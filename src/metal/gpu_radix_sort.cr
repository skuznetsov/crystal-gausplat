# GPU Radix Sort for Gaussian Splatting tile binning
# Sorts 64-bit keys (tile_id << 32 | depth) with associated gaussian IDs

require "./device"
require "./dispatch"
require "../core/buffer"

module GS
  module Metal
    # GPU Radix Sort module
    module GPURadixSort
      extend self

      # Load radix sort kernel source at compile time
      RADIX_SORT_SOURCE = {{ read_file("#{__DIR__}/kernels/radix_sort.metal") }}

      RADIX_BITS    = 4
      NUM_BUCKETS   = 16  # 2^4
      BLOCK_SIZE    = 256
      LOCAL_SORT_THRESHOLD = 512  # Use local sort for small arrays

      @@pipelines = Hash(String, ComputePipeline).new
      @@initialized = false

      # Lazy init
      def ensure_initialized
        return if @@initialized
        return unless Device.available?
        @@initialized = true
      end

      # Get or create pipeline
      def get_pipeline(name : String) : ComputePipeline
        @@pipelines[name] ||= ComputePipeline.new(name, RADIX_SORT_SOURCE, name)
      end

      # Check availability
      def available? : Bool
        Device.available?
      end

      # Sort 64-bit keys with associated 32-bit values
      # keys: uint64 buffer (tile_id << 32 | depth)
      # values: uint32 buffer (gaussian IDs)
      # n: number of elements
      def sort!(keys : MetalBuffer, values : MetalBuffer, n : Int32) : Nil
        return if n <= 1
        ensure_initialized
        return unless available?

        # Use local sort for small arrays
        if n <= LOCAL_SORT_THRESHOLD
          sort_local!(keys, values, n)
        else
          sort_global!(keys, values, n)
        end
      end

      # Sort small arrays entirely on GPU in one threadgroup
      private def sort_local!(keys : MetalBuffer, values : MetalBuffer, n : Int32) : Nil
        pipeline = get_pipeline("radix_sort_local")

        # Threadgroup memory sizes
        keys_mem = BLOCK_SIZE * 2 * 8  # uint64 = 8 bytes
        vals_mem = BLOCK_SIZE * 2 * 4  # uint32 = 4 bytes
        counts_mem = NUM_BUCKETS * 4
        offsets_mem = NUM_BUCKETS * 4

        Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(keys, 0)
          encoder.set_buffer(values, 1)
          encoder.set_value(n.to_u32, 2)
          encoder.set_threadgroup_memory(keys_mem, 0)
          encoder.set_threadgroup_memory(vals_mem, 1)
          encoder.set_threadgroup_memory(counts_mem, 2)
          encoder.set_threadgroup_memory(offsets_mem, 3)
          encoder.dispatch_1d(n, BLOCK_SIZE)
        end
      end

      # Sort large arrays with multiple passes
      private def sort_global!(keys : MetalBuffer, values : MetalBuffer, n : Int32) : Nil
        num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Allocate temp buffers
        keys_temp = MetalBuffer.new(n.to_i64 * 8)  # uint64
        values_temp = MetalBuffer.new(n.to_i64 * 4)  # uint32
        local_counts = MetalBuffer.new((num_blocks * NUM_BUCKETS).to_i64 * 4)
        block_sums = MetalBuffer.new(NUM_BUCKETS.to_i64 * 4)

        # Process 4 bits at a time, 16 passes for 64 bits
        keys_src = keys
        values_src = values
        keys_dst = keys_temp
        values_dst = values_temp

        16.times do |pass|
          bit_offset = (pass * RADIX_BITS).to_u32

          # Step 1: Count digits per block
          count_digits!(keys_src, local_counts, n, num_blocks, bit_offset)

          # Step 2: Prefix sum histograms
          prefix_sum_histograms!(local_counts, block_sums, num_blocks)

          # Step 3: Scatter
          scatter!(keys_src, values_src, keys_dst, values_dst, local_counts, n, bit_offset)

          # Swap buffers for next pass
          keys_src, keys_dst = keys_dst, keys_src
          values_src, values_dst = values_dst, values_src
        end

        # After 16 passes, result is in original buffers (even number of swaps)
        # If odd, need to copy back - but 16 is even, so we're good
      end

      private def count_digits!(
        keys : MetalBuffer,
        local_counts : MetalBuffer,
        n : Int32,
        num_blocks : Int32,
        bit_offset : UInt32
      ) : Nil
        pipeline = get_pipeline("radix_count_local")
        counts_mem = NUM_BUCKETS * 4

        Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(keys, 0)
          encoder.set_buffer(local_counts, 1)
          encoder.set_value(n.to_u32, 2)
          encoder.set_value(bit_offset, 3)
          encoder.set_threadgroup_memory(counts_mem, 0)
          encoder.dispatch_1d(n, BLOCK_SIZE)
        end
      end

      private def prefix_sum_histograms!(
        local_counts : MetalBuffer,
        block_sums : MetalBuffer,
        num_blocks : Int32
      ) : Nil
        # Step 1: Prefix sum each digit's histogram across blocks
        pipeline_sum = get_pipeline("radix_prefix_sum_histograms")
        shared_mem = 2 * BLOCK_SIZE * 4

        Dispatch.execute(pipeline_sum) do |encoder|
          encoder.set_buffer(local_counts, 0)
          encoder.set_buffer(block_sums, 1)
          encoder.set_value(num_blocks.to_u32, 2)
          encoder.set_threadgroup_memory(shared_mem, 0)
          # Dispatch 16 threadgroups (one per digit), each with BLOCK_SIZE threads
          # But we limit to num_blocks threads per group
          encoder.dispatch({Math.min(num_blocks, BLOCK_SIZE), NUM_BUCKETS, 1}, {BLOCK_SIZE, 1, 1})
        end

        # Step 2: Scan digit sums (16 elements, single thread)
        pipeline_scan = get_pipeline("radix_scan_digit_sums")

        Dispatch.execute(pipeline_scan) do |encoder|
          encoder.set_buffer(block_sums, 0)
          encoder.dispatch_1d(1, 1)
        end

        # Step 3: Add digit offsets to all histogram entries
        pipeline_add = get_pipeline("radix_add_digit_offsets")

        Dispatch.execute(pipeline_add) do |encoder|
          encoder.set_buffer(local_counts, 0)
          encoder.set_buffer(block_sums, 1)
          encoder.set_value(num_blocks.to_u32, 2)
          encoder.dispatch_1d(num_blocks * NUM_BUCKETS, BLOCK_SIZE)
        end
      end

      private def scatter!(
        keys_in : MetalBuffer,
        values_in : MetalBuffer,
        keys_out : MetalBuffer,
        values_out : MetalBuffer,
        local_counts : MetalBuffer,
        n : Int32,
        bit_offset : UInt32
      ) : Nil
        pipeline = get_pipeline("radix_scatter")
        offsets_mem = NUM_BUCKETS * 4
        counts_mem = NUM_BUCKETS * 4

        Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(keys_in, 0)
          encoder.set_buffer(values_in, 1)
          encoder.set_buffer(keys_out, 2)
          encoder.set_buffer(values_out, 3)
          encoder.set_buffer(local_counts, 4)
          encoder.set_value(n.to_u32, 5)
          encoder.set_value(bit_offset, 6)
          encoder.set_threadgroup_memory(offsets_mem, 0)
          encoder.set_threadgroup_memory(counts_mem, 1)
          encoder.dispatch_1d(n, BLOCK_SIZE)
        end
      end

      # Compute tile ranges after sorting
      def compute_tile_ranges!(
        sorted_keys : MetalBuffer,
        tile_ranges : MetalBuffer,
        n : Int32,
        num_tiles : Int32
      ) : Nil
        ensure_initialized

        # Initialize ranges to zero
        pipeline_init = get_pipeline("init_tile_ranges")

        Dispatch.execute(pipeline_init) do |encoder|
          encoder.set_buffer(tile_ranges, 0)
          encoder.set_value(num_tiles.to_u32, 1)
          encoder.dispatch_1d(num_tiles, BLOCK_SIZE)
        end

        return if n == 0

        # Find tile boundaries
        pipeline_ranges = get_pipeline("compute_tile_ranges")

        Dispatch.execute(pipeline_ranges) do |encoder|
          encoder.set_buffer(sorted_keys, 0)
          encoder.set_buffer(tile_ranges, 1)
          encoder.set_value(n.to_u32, 2)
          encoder.set_value(num_tiles.to_u32, 3)
          encoder.dispatch_1d(n, BLOCK_SIZE)
        end
      end
    end
  end
end
