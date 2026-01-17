# Metal Buffer wrapper with RAII semantics
# Manages GPU memory lifecycle, supports zero-copy on Apple Silicon

module GS
  # Storage mode for Metal buffers
  enum StorageMode
    Shared  # CPU + GPU accessible, zero-copy on Apple Silicon (default)
    Private # GPU only, fastest for GPU-exclusive data
    Managed # macOS only, explicit sync required
  end

  # Wraps a Metal buffer with automatic resource management
  # Uses MTLResourceStorageModeShared for unified memory on M-series chips
  class MetalBuffer
    getter size : Int64
    getter storage_mode : StorageMode
    getter? valid : Bool

    # Pointer to underlying MTLBuffer (void* in Crystal FFI)
    @handle : Pointer(Void)

    # Track if we own the buffer (vs wrapping external)
    @owned : Bool

    def initialize(@size : Int64, @storage_mode : StorageMode = StorageMode::Shared)
      raise ArgumentError.new("Buffer size must be positive") if @size <= 0
      @handle = MetalFFI.gs_create_buffer(@size, @storage_mode.value)
      @owned = true
      @valid = !@handle.null?
      raise "Failed to allocate Metal buffer of size #{@size}" unless @valid
    end

    # Wrap existing Array without copy (zero-copy)
    def self.from_array(data : Array(Float32), mode : StorageMode = StorageMode::Shared) : MetalBuffer
      byte_size = data.size.to_i64 * sizeof(Float32)
      buffer = new(byte_size, mode)
      buffer.write(data)
      buffer
    end

    # Wrap existing Slice
    def self.from_slice(data : Slice(Float32), mode : StorageMode = StorageMode::Shared) : MetalBuffer
      byte_size = data.size.to_i64 * sizeof(Float32)
      buffer = new(byte_size, mode)
      buffer.write_slice(data)
      buffer
    end

    # Write data to buffer
    def write(data : Array(Float32)) : Nil
      raise "Buffer invalid" unless @valid
      byte_size = data.size.to_i64 * sizeof(Float32)
      raise "Data size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      MetalFFI.gs_buffer_write(@handle, data.to_unsafe.as(Pointer(Void)), byte_size)
    end

    def write_slice(data : Slice(Float32)) : Nil
      raise "Buffer invalid" unless @valid
      byte_size = data.size.to_i64 * sizeof(Float32)
      raise "Data size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      MetalFFI.gs_buffer_write(@handle, data.to_unsafe.as(Pointer(Void)), byte_size)
    end

    # Read data from buffer
    def read(count : Int32) : Array(Float32)
      raise "Buffer invalid" unless @valid
      byte_size = count.to_i64 * sizeof(Float32)
      raise "Read size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      result = Array(Float32).new(count, 0.0_f32)
      MetalFFI.gs_buffer_read(@handle, result.to_unsafe.as(Pointer(Void)), byte_size)
      result
    end

    def read_to_slice(dest : Slice(Float32)) : Nil
      raise "Buffer invalid" unless @valid
      byte_size = dest.size.to_i64 * sizeof(Float32)
      raise "Read size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      MetalFFI.gs_buffer_read(@handle, dest.to_unsafe.as(Pointer(Void)), byte_size)
    end

    # Write raw bytes to buffer
    def write_bytes(ptr : Pointer(UInt8), byte_size : Int) : Nil
      raise "Buffer invalid" unless @valid
      raise "Data size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      MetalFFI.gs_buffer_write(@handle, ptr.as(Pointer(Void)), byte_size.to_i64)
    end

    # Read raw bytes from buffer
    def read_bytes(ptr : Pointer(UInt8), byte_size : Int) : Nil
      raise "Buffer invalid" unless @valid
      raise "Read size #{byte_size} exceeds buffer size #{@size}" if byte_size > @size
      MetalFFI.gs_buffer_read(@handle, ptr.as(Pointer(Void)), byte_size.to_i64)
    end

    # Get raw pointer for Metal kernel bindings
    def contents : Pointer(Void)
      raise "Buffer invalid" unless @valid
      MetalFFI.gs_buffer_contents(@handle)
    end

    # Get underlying handle for FFI
    def handle : Pointer(Void)
      @handle
    end

    # Element count (assuming Float32)
    def element_count : Int32
      (@size // sizeof(Float32)).to_i32
    end

    # Copy from another buffer
    def copy_from(src : MetalBuffer, byte_size : Int64) : Nil
      raise "Destination buffer invalid" unless @valid
      raise "Source buffer invalid" unless src.valid?
      raise "Copy size #{byte_size} exceeds destination buffer size #{@size}" if byte_size > @size
      raise "Copy size #{byte_size} exceeds source buffer size #{src.size}" if byte_size > src.size

      # On unified memory (Apple Silicon), we can directly copy via contents pointers
      MetalFFI.gs_buffer_copy(src.handle, self.handle, byte_size)
    end

    # Synchronize (for Managed mode on macOS)
    def sync : Nil
      return unless @valid && @storage_mode == StorageMode::Managed
      MetalFFI.gs_buffer_sync(@handle)
    end

    # RAII cleanup
    def finalize
      if @valid && @owned
        MetalFFI.gs_release_buffer(@handle)
        @valid = false
      end
    end

    # Manual release (for explicit control)
    def release : Nil
      if @valid && @owned
        MetalFFI.gs_release_buffer(@handle)
        @valid = false
      end
    end
  end

  # Buffer pool for reusing allocations (reduces Metal allocation overhead)
  class BufferPool
    ALIGNMENT = 256_i64 # Metal buffer alignment

    @pools : Hash(Int64, Array(MetalBuffer))
    @max_cached : Int32

    def initialize(@max_cached : Int32 = 32)
      @pools = Hash(Int64, Array(MetalBuffer)).new
    end

    # Get buffer of at least `size` bytes, reusing if possible
    def acquire(size : Int64, mode : StorageMode = StorageMode::Shared) : MetalBuffer
      aligned_size = align_size(size)

      if pool = @pools[aligned_size]?
        if buffer = pool.pop?
          return buffer if buffer.valid?
        end
      end

      MetalBuffer.new(aligned_size, mode)
    end

    # Return buffer to pool for reuse
    def release(buffer : MetalBuffer) : Nil
      return unless buffer.valid?

      aligned_size = align_size(buffer.size)
      pool = @pools[aligned_size] ||= Array(MetalBuffer).new

      if pool.size < @max_cached
        pool << buffer
      else
        buffer.release
      end
    end

    # Clear all cached buffers
    def clear : Nil
      @pools.each_value do |pool|
        pool.each(&.release)
        pool.clear
      end
    end

    private def align_size(size : Int64) : Int64
      ((size + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    end
  end

  # Global buffer pool instance
  class_getter buffer_pool : BufferPool = BufferPool.new
end

# Metal FFI declarations (implemented in metal_bridge.mm)
{% if flag?(:darwin) %}
@[Link(ldflags: "-framework Metal -framework Foundation")]
lib MetalFFI
  fun gs_create_buffer = gs_create_buffer(size : Int64, storage_mode : Int32) : Pointer(Void)
  fun gs_release_buffer = gs_release_buffer(handle : Pointer(Void)) : Void
  fun gs_buffer_contents = gs_buffer_contents(handle : Pointer(Void)) : Pointer(Void)
  fun gs_buffer_write = gs_buffer_write(handle : Pointer(Void), data : Pointer(Void), size : Int64) : Void
  fun gs_buffer_read = gs_buffer_read(handle : Pointer(Void), dest : Pointer(Void), size : Int64) : Void
  fun gs_buffer_sync = gs_buffer_sync(handle : Pointer(Void)) : Void
  fun gs_buffer_copy = gs_buffer_copy(src : Pointer(Void), dst : Pointer(Void), size : Int64) : Void
end
{% else %}
# Stub for non-Darwin platforms (CPU fallback)
lib MetalFFI
  fun gs_create_buffer = gs_create_buffer(size : Int64, storage_mode : Int32) : Pointer(Void)
  fun gs_release_buffer = gs_release_buffer(handle : Pointer(Void)) : Void
  fun gs_buffer_contents = gs_buffer_contents(handle : Pointer(Void)) : Pointer(Void)
  fun gs_buffer_write = gs_buffer_write(handle : Pointer(Void), data : Pointer(Void), size : Int64) : Void
  fun gs_buffer_read = gs_buffer_read(handle : Pointer(Void), dest : Pointer(Void), size : Int64) : Void
  fun gs_buffer_sync = gs_buffer_sync(handle : Pointer(Void)) : Void
  fun gs_buffer_copy = gs_buffer_copy(src : Pointer(Void), dst : Pointer(Void), size : Int64) : Void
end
{% end %}
