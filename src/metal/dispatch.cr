# Kernel dispatch utilities for Metal compute shaders
# Handles grid/threadgroup sizing and buffer binding

require "./device"
require "../core/tensor"

module GS
  module Metal
    # Compute encoder for setting up kernel dispatch
    struct ComputeEncoder
      @cmd_buffer : Pointer(Void)
      @encoder : Pointer(Void)
      @pipeline : ComputePipeline?

      def initialize(command_buffer : CommandBuffer)
        @cmd_buffer = command_buffer.handle
        @encoder = MetalDispatchFFI.create_compute_encoder(@cmd_buffer)
        raise "Failed to create compute encoder" if @encoder.null?
        @pipeline = nil
      end

      def set_pipeline(pipeline : ComputePipeline) : self
        @pipeline = pipeline
        MetalDispatchFFI.encoder_set_pipeline(@encoder, pipeline.handle)
        self
      end

      # Bind Metal buffer at index
      def set_buffer(buffer : MetalBuffer, index : Int32, offset : Int64 = 0) : self
        MetalDispatchFFI.encoder_set_buffer(@encoder, buffer.handle, offset, index)
        self
      end

      # Bind tensor's underlying buffer
      def set_tensor(tensor : Tensor, index : Int32) : self
        raise "Tensor must be on GPU" unless tensor.on_gpu?
        set_buffer(tensor.buffer.not_nil!, index)
      end

      # Bind raw bytes (for small constants)
      def set_bytes(data : Pointer(Void), length : Int32, index : Int32) : self
        MetalDispatchFFI.encoder_set_bytes(@encoder, data, length, index)
        self
      end

      # Bind scalar value
      def set_value(value : Float32, index : Int32) : self
        ptr = pointerof(value).as(Pointer(Void))
        set_bytes(ptr, sizeof(Float32), index)
      end

      def set_value(value : Int32, index : Int32) : self
        ptr = pointerof(value).as(Pointer(Void))
        set_bytes(ptr, sizeof(Int32), index)
      end

      def set_value(value : UInt32, index : Int32) : self
        ptr = pointerof(value).as(Pointer(Void))
        set_bytes(ptr, sizeof(UInt32), index)
      end

      # Set float3 (as StaticArray(Float32, 3))
      def set_value(value : StaticArray(Float32, 3), index : Int32) : self
        ptr = value.to_unsafe.as(Pointer(Void))
        set_bytes(ptr, sizeof(Float32) * 3, index)
      end

      # Set float4 (as StaticArray(Float32, 4))
      def set_value(value : StaticArray(Float32, 4), index : Int32) : self
        ptr = value.to_unsafe.as(Pointer(Void))
        set_bytes(ptr, sizeof(Float32) * 4, index)
      end

      # Dispatch with explicit grid/threadgroup sizes
      def dispatch(grid_size : {Int32, Int32, Int32}, threadgroup_size : {Int32, Int32, Int32}) : self
        MetalDispatchFFI.encoder_dispatch_threads(
          @encoder,
          grid_size[0], grid_size[1], grid_size[2],
          threadgroup_size[0], threadgroup_size[1], threadgroup_size[2]
        )
        self
      end

      # Dispatch 1D workload
      def dispatch_1d(count : Int32, threadgroup_size : Int32 = 256) : self
        # Clamp threadgroup size to pipeline max
        max_threads = @pipeline.try(&.max_total_threads_per_threadgroup) || 1024
        tg_size = Math.min(threadgroup_size, max_threads)

        # Calculate grid size to cover all elements
        grid_width = count

        dispatch({grid_width, 1, 1}, {tg_size, 1, 1})
      end

      # Dispatch 2D workload (e.g., images)
      def dispatch_2d(width : Int32, height : Int32, threadgroup_size : {Int32, Int32} = {16, 16}) : self
        max_threads = @pipeline.try(&.max_total_threads_per_threadgroup) || 1024
        tg_w = Math.min(threadgroup_size[0], max_threads)
        tg_h = Math.min(threadgroup_size[1], max_threads // tg_w)

        dispatch({width, height, 1}, {tg_w, tg_h, 1})
      end

      # Dispatch 3D workload
      def dispatch_3d(width : Int32, height : Int32, depth : Int32, threadgroup_size : {Int32, Int32, Int32} = {8, 8, 8}) : self
        max_threads = @pipeline.try(&.max_total_threads_per_threadgroup) || 1024
        tg_w = Math.min(threadgroup_size[0], max_threads)
        tg_h = Math.min(threadgroup_size[1], max_threads // tg_w)
        tg_d = Math.min(threadgroup_size[2], max_threads // (tg_w * tg_h))

        dispatch({width, height, depth}, {tg_w, tg_h, tg_d})
      end

      # Set threadgroup memory (for kernels using threadgroup storage)
      def set_threadgroup_memory(length : Int32, index : Int32) : self
        MetalDispatchFFI.encoder_set_threadgroup_memory(@encoder, length, index)
        self
      end

      # End encoding
      def end_encoding : Nil
        MetalDispatchFFI.encoder_end_encoding(@encoder)
      end
    end

    # High-level dispatch helper
    module Dispatch
      extend self

      # Execute a kernel synchronously
      def execute(pipeline : ComputePipeline, &block : ComputeEncoder -> Nil) : Nil
        cmd_buffer = CommandBuffer.new
        encoder = ComputeEncoder.new(cmd_buffer)
        encoder.set_pipeline(pipeline)

        yield encoder

        encoder.end_encoding
        cmd_buffer.commit_and_wait
      end

      # Execute multiple kernels in sequence (single command buffer)
      def execute_sequence(&block : CommandBuffer -> Nil) : Nil
        cmd_buffer = CommandBuffer.new
        yield cmd_buffer
        cmd_buffer.commit_and_wait
      end

      # Execute kernel asynchronously
      def execute_async(pipeline : ComputePipeline, &block : ComputeEncoder -> Nil) : CommandBuffer
        cmd_buffer = CommandBuffer.new
        encoder = ComputeEncoder.new(cmd_buffer)
        encoder.set_pipeline(pipeline)

        yield encoder

        encoder.end_encoding
        cmd_buffer.commit
        cmd_buffer
      end

      # Optimal threadgroup size for 1D workloads
      def optimal_threadgroup_1d(count : Int32, pipeline : ComputePipeline? = nil) : Int32
        max_threads = pipeline.try(&.max_total_threads_per_threadgroup) || Device.instance.max_threads_per_threadgroup
        # Use power of 2 size, typically 256 is good for most workloads
        size = 256
        while size > max_threads
          size //= 2
        end
        size
      end

      # Optimal threadgroup size for 2D workloads (images, tiles)
      def optimal_threadgroup_2d(width : Int32, height : Int32, pipeline : ComputePipeline? = nil) : {Int32, Int32}
        max_threads = pipeline.try(&.max_total_threads_per_threadgroup) || Device.instance.max_threads_per_threadgroup
        # 16x16 = 256 threads is typical for image processing
        w = 16
        h = 16
        while w * h > max_threads
          if w >= h
            w //= 2
          else
            h //= 2
          end
        end
        {w, h}
      end
    end

    # Common kernel parameter structs
    module KernelParams
      # Elementwise operation parameters
      struct ElementwiseParams
        property count : UInt32
        property alpha : Float32 # Scalar multiplier (optional)
        property beta : Float32  # Scalar addend (optional)

        def initialize(@count : UInt32, @alpha : Float32 = 1.0_f32, @beta : Float32 = 0.0_f32)
        end
      end

      # Matrix multiplication parameters
      struct MatmulParams
        property m : UInt32 # Rows of A and C
        property n : UInt32 # Cols of B and C
        property k : UInt32 # Cols of A, Rows of B
        property alpha : Float32
        property beta : Float32

        def initialize(@m : UInt32, @n : UInt32, @k : UInt32, @alpha : Float32 = 1.0_f32, @beta : Float32 = 0.0_f32)
        end
      end

      # Reduction parameters
      struct ReductionParams
        property count : UInt32
        property stride : UInt32 # For strided reductions

        def initialize(@count : UInt32, @stride : UInt32 = 1)
        end
      end

      # 2D grid parameters (for images, tiles)
      struct Grid2DParams
        property width : UInt32
        property height : UInt32
        property channels : UInt32

        def initialize(@width : UInt32, @height : UInt32, @channels : UInt32 = 1)
        end
      end
    end
  end
end

# Metal Dispatch FFI declarations
{% if flag?(:darwin) %}
@[Link(ldflags: "-framework Metal -framework Foundation")]
lib MetalDispatchFFI
  # Compute encoder
  fun create_compute_encoder = gs_create_compute_encoder(cmd_buffer : Pointer(Void)) : Pointer(Void)
  fun encoder_set_pipeline = gs_encoder_set_pipeline(encoder : Pointer(Void), pipeline : Pointer(Void)) : Void
  fun encoder_set_buffer = gs_encoder_set_buffer(encoder : Pointer(Void), buffer : Pointer(Void), offset : Int64, index : Int32) : Void
  fun encoder_set_bytes = gs_encoder_set_bytes(encoder : Pointer(Void), data : Pointer(Void), length : Int32, index : Int32) : Void
  fun encoder_dispatch_threads = gs_encoder_dispatch_threads(
    encoder : Pointer(Void),
    grid_x : Int32, grid_y : Int32, grid_z : Int32,
    tg_x : Int32, tg_y : Int32, tg_z : Int32
  ) : Void
  fun encoder_end_encoding = gs_encoder_end_encoding(encoder : Pointer(Void)) : Void
  fun encoder_set_threadgroup_memory = gs_encoder_set_threadgroup_memory(encoder : Pointer(Void), length : Int32, index : Int32) : Void
end
{% else %}
lib MetalDispatchFFI
  fun create_compute_encoder = gs_create_compute_encoder(cmd_buffer : Pointer(Void)) : Pointer(Void)
  fun encoder_set_pipeline = gs_encoder_set_pipeline(encoder : Pointer(Void), pipeline : Pointer(Void)) : Void
  fun encoder_set_buffer = gs_encoder_set_buffer(encoder : Pointer(Void), buffer : Pointer(Void), offset : Int64, index : Int32) : Void
  fun encoder_set_bytes = gs_encoder_set_bytes(encoder : Pointer(Void), data : Pointer(Void), length : Int32, index : Int32) : Void
  fun encoder_dispatch_threads = gs_encoder_dispatch_threads(
    encoder : Pointer(Void),
    grid_x : Int32, grid_y : Int32, grid_z : Int32,
    tg_x : Int32, tg_y : Int32, tg_z : Int32
  ) : Void
  fun encoder_end_encoding = gs_encoder_end_encoding(encoder : Pointer(Void)) : Void
  fun encoder_set_threadgroup_memory = gs_encoder_set_threadgroup_memory(encoder : Pointer(Void), length : Int32, index : Int32) : Void
end
{% end %}
