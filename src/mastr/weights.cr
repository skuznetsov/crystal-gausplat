# Safetensors format loader for MASt3R weights
# Format: [8-byte header size] [JSON header] [tensor data]

require "json"
require "../core/tensor"

module GS
  module MASt3R
    # Tensor metadata from safetensors header
    struct TensorInfo
      property name : String
      property dtype : String
      property shape : Array(Int64)
      property data_start : Int64
      property data_end : Int64

      def initialize(@name, @dtype, @shape, @data_start, @data_end)
      end

      def byte_size : Int64
        @data_end - @data_start
      end

      def numel : Int64
        @shape.reduce(1_i64) { |a, b| a * b }
      end
    end

    # Safetensors file reader
    class SafetensorsLoader
      getter path : String
      getter tensors : Hash(String, TensorInfo)
      @header_size : Int64
      @data_offset : Int64

      def initialize(@path : String)
        @tensors = Hash(String, TensorInfo).new
        @header_size = 0_i64
        @data_offset = 0_i64
        parse_header!
      end

      # List all tensor names
      def tensor_names : Array(String)
        @tensors.keys
      end

      # Get tensor info by name
      def tensor_info(name : String) : TensorInfo?
        @tensors[name]?
      end

      # Load a single tensor by name
      def load_tensor(name : String, device : Tensor::Device = Tensor::Device::CPU) : Tensor
        info = @tensors[name]?
        raise "Tensor '#{name}' not found in #{@path}" unless info

        load_tensor_data(info, device)
      end

      # Load multiple tensors by prefix (e.g., "encoder.layer.0")
      def load_by_prefix(prefix : String, device : Tensor::Device = Tensor::Device::CPU) : Hash(String, Tensor)
        result = Hash(String, Tensor).new

        @tensors.each do |name, info|
          if name.starts_with?(prefix)
            result[name] = load_tensor_data(info, device)
          end
        end

        result
      end

      # Load all tensors
      def load_all(device : Tensor::Device = Tensor::Device::CPU) : Hash(String, Tensor)
        result = Hash(String, Tensor).new

        @tensors.each do |name, info|
          result[name] = load_tensor_data(info, device)
        end

        result
      end

      # Parse the safetensors header
      private def parse_header!
        File.open(@path, "rb") do |file|
          # Read header size (8 bytes, little-endian)
          header_size_bytes = Bytes.new(8)
          file.read_fully(header_size_bytes)
          @header_size = IO::ByteFormat::LittleEndian.decode(Int64, header_size_bytes)

          # Read JSON header
          header_bytes = Bytes.new(@header_size)
          file.read_fully(header_bytes)
          header_json = String.new(header_bytes)

          # Data starts after header
          @data_offset = 8_i64 + @header_size

          # Parse JSON
          parsed = JSON.parse(header_json)

          # Extract tensor metadata
          parsed.as_h.each do |key, value|
            next if key == "__metadata__"

            tensor_data = value.as_h
            dtype = tensor_data["dtype"].as_s
            shape = tensor_data["shape"].as_a.map(&.as_i64)
            offsets = tensor_data["data_offsets"].as_a
            data_start = offsets[0].as_i64
            data_end = offsets[1].as_i64

            @tensors[key] = TensorInfo.new(key, dtype, shape, data_start, data_end)
          end
        end
      end

      # Load tensor data from file
      private def load_tensor_data(info : TensorInfo, device : Tensor::Device) : Tensor
        # Convert shape to Int32
        shape = Shape.new(info.shape.map(&.to_i32))

        # Determine element type and size
        elem_size = dtype_size(info.dtype)
        expected_bytes = info.numel * elem_size

        raise "Tensor size mismatch: expected #{expected_bytes}, got #{info.byte_size}" unless expected_bytes == info.byte_size

        # Read raw bytes
        bytes = Bytes.new(info.byte_size)
        File.open(@path, "rb") do |file|
          file.seek(@data_offset + info.data_start)
          file.read_fully(bytes)
        end

        # Convert to Float32 array
        data = case info.dtype
               when "F32"
                 bytes_to_f32_array(bytes)
               when "F16"
                 bytes_to_f16_array(bytes)
               when "BF16"
                 bytes_to_bf16_array(bytes)
               else
                 raise "Unsupported dtype: #{info.dtype}"
               end

        # Create tensor
        tensor = Tensor.from_array(data, shape)
        device.gpu? ? tensor.to_gpu : tensor
      end

      private def dtype_size(dtype : String) : Int64
        case dtype
        when "F32", "I32", "U32" then 4_i64
        when "F16", "BF16", "I16", "U16" then 2_i64
        when "F64", "I64", "U64" then 8_i64
        when "I8", "U8", "BOOL" then 1_i64
        else raise "Unknown dtype: #{dtype}"
        end
      end

      private def bytes_to_f32_array(bytes : Bytes) : Array(Float32)
        count = bytes.size // 4
        result = Array(Float32).new(count)
        count.times do |i|
          value = IO::ByteFormat::LittleEndian.decode(Float32, bytes[i * 4, 4])
          result << value
        end
        result
      end

      private def bytes_to_f16_array(bytes : Bytes) : Array(Float32)
        # Convert FP16 to FP32
        count = bytes.size // 2
        result = Array(Float32).new(count)
        count.times do |i|
          half = IO::ByteFormat::LittleEndian.decode(UInt16, bytes[i * 2, 2])
          result << fp16_to_fp32(half)
        end
        result
      end

      private def bytes_to_bf16_array(bytes : Bytes) : Array(Float32)
        # Convert BF16 to FP32 (just shift left 16 bits)
        count = bytes.size // 2
        result = Array(Float32).new(count)
        count.times do |i|
          bf16 = IO::ByteFormat::LittleEndian.decode(UInt16, bytes[i * 2, 2])
          # BF16 is just truncated FP32, so shift to upper bits
          f32_bits = bf16.to_u32 << 16
          result << f32_bits.unsafe_as(Float32)
        end
        result
      end

      # IEEE 754 FP16 to FP32 conversion
      private def fp16_to_fp32(half : UInt16) : Float32
        sign = (half >> 15) & 0x1_u32
        exponent = (half >> 10) & 0x1f_u32
        mantissa = half & 0x3ff_u32

        if exponent == 0
          if mantissa == 0
            # Zero
            f32_bits = sign << 31
          else
            # Subnormal
            while (mantissa & 0x400_u32) == 0
              mantissa <<= 1
              exponent -= 1
            end
            exponent += 1
            mantissa &= ~0x400_u32
            f32_bits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13)
          end
        elsif exponent == 31
          # Inf or NaN
          f32_bits = (sign << 31) | 0x7f800000_u32 | (mantissa << 13)
        else
          # Normal
          f32_bits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13)
        end

        f32_bits.unsafe_as(Float32)
      end
    end

    # Convenience function
    def self.load_safetensors(path : String) : SafetensorsLoader
      SafetensorsLoader.new(path)
    end
  end
end
