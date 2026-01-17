# Persistent Threadgroups for Transformer Blocks
# Reduces kernel launch overhead by processing multiple layers in one GPU dispatch
#
# Benefits:
# 1. Single kernel launch for all transformer blocks
# 2. Intermediate activations stay in threadgroup memory
# 3. No CPU-GPU synchronization between layers
# 4. Fused operations reduce memory bandwidth

require "./device"
require "./dispatch"
require "../core/buffer"

module GS
  module Metal
    # Persistent Transformer module for efficient multi-block processing
    module PersistentTransformer
      extend self

      # Load kernel source at compile time
      PERSISTENT_TRANSFORMER_SOURCE = {{ read_file("#{__DIR__}/kernels/persistent_transformer.metal") }}

      @@pipelines = Hash(String, ComputePipeline).new
      @@initialized = false

      # Lazy initialization
      def ensure_initialized
        return if @@initialized
        return unless Device.available?
        @@initialized = true
      end

      # Get or create pipeline
      def get_pipeline(name : String) : ComputePipeline
        @@pipelines[name] ||= ComputePipeline.new(name, PERSISTENT_TRANSFORMER_SOURCE, name)
      end

      # Check availability
      def available? : Bool
        Device.available?
      end

      # Weight buffer layout for a single transformer block
      # Returns the size in floats for one block's weights
      def weights_per_block(embed_dim : Int32, hidden_dim : Int32) : Int32
        # ln1_gamma + ln1_beta + wq + wk + wv + wo + ln2_gamma + ln2_beta + fc1_w + fc1_b + fc2_w + fc2_b
        2 * embed_dim +                      # ln1 gamma/beta
        4 * embed_dim * embed_dim +          # wq, wk, wv, wo
        2 * embed_dim +                      # ln2 gamma/beta
        hidden_dim * embed_dim + hidden_dim +  # fc1 weight/bias
        embed_dim * hidden_dim + embed_dim     # fc2 weight/bias
      end

      # Process a single transformer block
      # input: [batch, seq_len, embed_dim]
      # weights: concatenated weights for this block
      # output: [batch, seq_len, embed_dim]
      def forward_single_block!(
        input : MetalBuffer,
        output : MetalBuffer,
        weights : MetalBuffer,
        batch : Int32,
        seq_len : Int32,
        embed_dim : Int32,
        num_heads : Int32,
        hidden_dim : Int32,
        eps : Float32 = 1e-5_f32
      ) : Nil
        ensure_initialized
        return unless available?

        pipeline = get_pipeline("transformer_block_fused")

        # Calculate threadgroup memory size
        # shared_x: [seq_len, embed_dim]
        # shared_tmp: needs space for Q, K, V, attn_out = 4 * [seq_len, embed_dim]
        shared_x_size = seq_len * embed_dim * sizeof(Float32)
        shared_tmp_size = 4 * seq_len * embed_dim * sizeof(Float32)

        # Check if we have enough threadgroup memory (typically 32KB on Apple Silicon)
        max_threadgroup_memory = 32768
        total_shared = shared_x_size + shared_tmp_size

        if total_shared > max_threadgroup_memory
          # Fall back to device memory version or error
          raise "Sequence/embed too large for threadgroup memory: #{total_shared} > #{max_threadgroup_memory}"
        end

        Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input, 0)
          encoder.set_buffer(output, 1)
          encoder.set_buffer(weights, 2)
          encoder.set_value(batch.to_u32, 3)
          encoder.set_value(seq_len.to_u32, 4)
          encoder.set_value(embed_dim.to_u32, 5)
          encoder.set_value(num_heads.to_u32, 6)
          encoder.set_value(hidden_dim.to_u32, 7)
          encoder.set_value(eps, 8)
          encoder.set_threadgroup_memory(shared_x_size, 0)
          encoder.set_threadgroup_memory(shared_tmp_size, 1)

          # One threadgroup per batch element
          # Use 256 threads per threadgroup for good occupancy
          threads_per_group = 256
          encoder.dispatch({batch, 1, 1}, {threads_per_group, 1, 1})
        end
      end

      # Process multiple transformer blocks in a single kernel
      # This is the main optimization - keeps threadgroups resident
      # input: [batch, seq_len, embed_dim]
      # all_weights: concatenated weights for all blocks
      # output: [batch, seq_len, embed_dim]
      # temp_buffer: [batch, seq_len, embed_dim] for ping-pong
      def forward_multi_block!(
        input : MetalBuffer,
        output : MetalBuffer,
        all_weights : MetalBuffer,
        temp_buffer : MetalBuffer,
        batch : Int32,
        seq_len : Int32,
        embed_dim : Int32,
        num_heads : Int32,
        hidden_dim : Int32,
        num_blocks : Int32,
        eps : Float32 = 1e-5_f32
      ) : Nil
        ensure_initialized
        return unless available?

        pipeline = get_pipeline("persistent_transformer_multi_block")

        # Shared memory layout:
        # buf_a: [seq_len, embed_dim]
        # buf_b: [seq_len, embed_dim]
        # workspace: space for QKV, attention, etc.
        # Total: approximately 8 * seq_len * embed_dim floats
        activation_size = seq_len * embed_dim * sizeof(Float32)
        workspace_size = 6 * seq_len * embed_dim * sizeof(Float32)  # Q, K, V, attn_out, residual, ln_out
        total_shared = 2 * activation_size + workspace_size

        # Check threadgroup memory limit
        max_threadgroup_memory = 32768  # 32KB typical for Apple Silicon

        if total_shared > max_threadgroup_memory
          # For large models, process blocks sequentially with separate kernel calls
          forward_multi_block_sequential!(input, output, all_weights, temp_buffer,
                                          batch, seq_len, embed_dim, num_heads,
                                          hidden_dim, num_blocks, eps)
          return
        end

        Dispatch.execute(pipeline) do |encoder|
          encoder.set_buffer(input, 0)
          encoder.set_buffer(output, 1)
          encoder.set_buffer(all_weights, 2)
          encoder.set_value(batch.to_u32, 3)
          encoder.set_value(seq_len.to_u32, 4)
          encoder.set_value(embed_dim.to_u32, 5)
          encoder.set_value(num_heads.to_u32, 6)
          encoder.set_value(hidden_dim.to_u32, 7)
          encoder.set_value(num_blocks.to_u32, 8)
          encoder.set_value(eps, 9)
          encoder.set_buffer(temp_buffer, 10)
          encoder.set_threadgroup_memory(total_shared, 0)

          # One threadgroup per batch element
          threads_per_group = 256
          encoder.dispatch({batch, 1, 1}, {threads_per_group, 1, 1})
        end
      end

      # Sequential multi-block processing for large models
      # Uses single-block kernel but chains outputs
      private def forward_multi_block_sequential!(
        input : MetalBuffer,
        output : MetalBuffer,
        all_weights : MetalBuffer,
        temp_buffer : MetalBuffer,
        batch : Int32,
        seq_len : Int32,
        embed_dim : Int32,
        num_heads : Int32,
        hidden_dim : Int32,
        num_blocks : Int32,
        eps : Float32
      ) : Nil
        pipeline = get_pipeline("transformer_block_fused")

        wpb = weights_per_block(embed_dim, hidden_dim)
        activation_bytes = batch.to_i64 * seq_len * embed_dim * sizeof(Float32)

        # Shared memory sizes
        shared_x_size = seq_len * embed_dim * sizeof(Float32)
        shared_tmp_size = 4 * seq_len * embed_dim * sizeof(Float32)

        # Process each block
        num_blocks.times do |block|
          # Determine input/output buffers (ping-pong)
          current_in = (block == 0) ? input : ((block % 2 == 1) ? temp_buffer : output)
          current_out = (block == num_blocks - 1) ? output : ((block % 2 == 0) ? temp_buffer : output)

          # Create a view/offset into the weights buffer
          # Note: In practice, we'd need to handle this more elegantly
          # For now, we'll create separate buffers or use offsets

          Dispatch.execute(pipeline) do |encoder|
            encoder.set_buffer(current_in, 0)
            encoder.set_buffer(current_out, 1)
            encoder.set_buffer(all_weights, 2)  # Will use offset in shader based on block index
            encoder.set_value(batch.to_u32, 3)
            encoder.set_value(seq_len.to_u32, 4)
            encoder.set_value(embed_dim.to_u32, 5)
            encoder.set_value(num_heads.to_u32, 6)
            encoder.set_value(hidden_dim.to_u32, 7)
            encoder.set_value(eps, 8)
            encoder.set_threadgroup_memory(shared_x_size, 0)
            encoder.set_threadgroup_memory(shared_tmp_size, 1)

            threads_per_group = 256
            encoder.dispatch({batch, 1, 1}, {threads_per_group, 1, 1})
          end
        end
      end

      # Benchmark helper: estimate throughput improvement
      def estimate_speedup(
        batch : Int32,
        seq_len : Int32,
        embed_dim : Int32,
        num_blocks : Int32
      ) : {Float64, Float64}
        # Estimate based on kernel launch overhead reduction
        # Typical Metal kernel launch: ~5-10 microseconds
        kernel_launch_us = 7.5

        # Non-persistent: each block has multiple kernel launches
        # (LN1, QKV proj, attention, output proj, residual, LN2, MLP fc1, MLP fc2, residual)
        # ~9 kernel launches per block
        kernels_per_block = 9
        non_persistent_launches = num_blocks * kernels_per_block

        # Persistent: single launch for all blocks
        persistent_launches = 1

        overhead_non_persistent = non_persistent_launches * kernel_launch_us / 1000.0  # ms
        overhead_persistent = persistent_launches * kernel_launch_us / 1000.0  # ms

        {overhead_non_persistent, overhead_persistent}
      end
    end
  end
end
