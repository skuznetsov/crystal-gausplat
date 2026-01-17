# Image I/O utilities
# Loads PNG/JPEG images using macOS ImageIO framework

require "../core/tensor"

module GS
  module Utils
    # FFI bindings for macOS ImageIO
    @[Link(framework: "CoreGraphics")]
    @[Link(framework: "ImageIO")]
    @[Link(framework: "CoreFoundation")]
    lib ImageIOLib
      # CoreFoundation types
      type CFDataRef = Void*
      type CFURLRef = Void*
      type CFStringRef = Void*
      type CFDictionaryRef = Void*
      type CFAllocatorRef = Void*

      # CoreGraphics types
      type CGImageRef = Void*
      type CGColorSpaceRef = Void*
      type CGContextRef = Void*
      type CGDataProviderRef = Void*

      # ImageIO types
      type CGImageSourceRef = Void*

      # CoreFoundation functions
      fun CFRelease(cf : Void*)
      fun CFDataCreate(allocator : CFAllocatorRef, bytes : UInt8*, length : Int64) : CFDataRef
      fun CFURLCreateWithFileSystemPath(allocator : CFAllocatorRef, path : CFStringRef, style : Int32, isDir : Bool) : CFURLRef
      fun CFStringCreateWithCString(allocator : CFAllocatorRef, str : UInt8*, encoding : UInt32) : CFStringRef

      # ImageIO functions
      fun CGImageSourceCreateWithData(data : CFDataRef, options : CFDictionaryRef) : CGImageSourceRef
      fun CGImageSourceCreateWithURL(url : CFURLRef, options : CFDictionaryRef) : CGImageSourceRef
      fun CGImageSourceCreateImageAtIndex(source : CGImageSourceRef, index : LibC::SizeT, options : CFDictionaryRef) : CGImageRef

      # CoreGraphics functions
      fun CGImageGetWidth(image : CGImageRef) : LibC::SizeT
      fun CGImageGetHeight(image : CGImageRef) : LibC::SizeT
      fun CGImageGetBitsPerComponent(image : CGImageRef) : LibC::SizeT
      fun CGImageGetBitsPerPixel(image : CGImageRef) : LibC::SizeT
      fun CGImageGetBytesPerRow(image : CGImageRef) : LibC::SizeT
      fun CGImageRelease(image : CGImageRef)

      fun CGColorSpaceCreateDeviceRGB : CGColorSpaceRef
      fun CGColorSpaceRelease(space : CGColorSpaceRef)

      fun CGBitmapContextCreate(
        data : Void*,
        width : LibC::SizeT,
        height : LibC::SizeT,
        bitsPerComponent : LibC::SizeT,
        bytesPerRow : LibC::SizeT,
        space : CGColorSpaceRef,
        bitmapInfo : UInt32
      ) : CGContextRef
      fun CGContextRelease(context : CGContextRef)
      fun CGContextDrawImage(context : CGContextRef, rect : CGRect, image : CGImageRef)

      # CGRect struct
      struct CGRect
        x : Float64
        y : Float64
        width : Float64
        height : Float64
      end
    end

    # Image class for loaded images
    class Image
      getter width : Int32
      getter height : Int32
      getter channels : Int32
      getter data : Array(UInt8)

      def initialize(@width, @height, @channels, @data)
      end

      # Convert to tensor [height, width, channels] normalized to 0-1
      def to_tensor(device : Tensor::Device = Tensor::Device::CPU) : Tensor
        tensor = Tensor.new(@height, @width, @channels, device: Tensor::Device::CPU)
        tensor_data = tensor.cpu_data.not_nil!

        @height.times do |y|
          @width.times do |x|
            @channels.times do |c|
              idx = (y * @width + x) * @channels + c
              tensor_idx = y * @width * @channels + x * @channels + c
              tensor_data[tensor_idx] = @data[idx].to_f32 / 255.0_f32
            end
          end
        end

        device.gpu? ? tensor.to_gpu : tensor
      end

      # Convert to tensor [channels, height, width] (CHW format) normalized to 0-1
      def to_tensor_chw(device : Tensor::Device = Tensor::Device::CPU) : Tensor
        tensor = Tensor.new(@channels, @height, @width, device: Tensor::Device::CPU)
        tensor_data = tensor.cpu_data.not_nil!

        @channels.times do |c|
          @height.times do |y|
            @width.times do |x|
              src_idx = (y * @width + x) * @channels + c
              dst_idx = c * @height * @width + y * @width + x
              tensor_data[dst_idx] = @data[src_idx].to_f32 / 255.0_f32
            end
          end
        end

        device.gpu? ? tensor.to_gpu : tensor
      end

      # Convert to tensor with batch dimension [1, channels, height, width]
      def to_tensor_bchw(device : Tensor::Device = Tensor::Device::CPU) : Tensor
        tensor = Tensor.new(1, @channels, @height, @width, device: Tensor::Device::CPU)
        tensor_data = tensor.cpu_data.not_nil!

        @channels.times do |c|
          @height.times do |y|
            @width.times do |x|
              src_idx = (y * @width + x) * @channels + c
              dst_idx = c * @height * @width + y * @width + x
              tensor_data[dst_idx] = @data[src_idx].to_f32 / 255.0_f32
            end
          end
        end

        device.gpu? ? tensor.to_gpu : tensor
      end
    end

    # Image loader using macOS ImageIO
    module ImageIO
      # kCFStringEncodingUTF8
      UTF8_ENCODING = 0x08000100_u32

      # kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
      RGBA_BITMAP_INFO = 1_u32 | (2_u32 << 12)

      # kCFURLPOSIXPathStyle
      POSIX_PATH_STYLE = 0_i32

      # Load image from file path
      def self.load(path : String) : Image
        # Create CFString for path
        path_cf = ImageIOLib.CFStringCreateWithCString(
          Pointer(Void).null,
          path.to_unsafe,
          UTF8_ENCODING
        )
        raise "Failed to create CFString for path" if path_cf.null?

        # Create CFURL
        url = ImageIOLib.CFURLCreateWithFileSystemPath(
          Pointer(Void).null,
          path_cf,
          POSIX_PATH_STYLE,
          false
        )
        ImageIOLib.CFRelease(path_cf)
        raise "Failed to create CFURL for: #{path}" if url.null?

        # Create image source
        source = ImageIOLib.CGImageSourceCreateWithURL(url, Pointer(Void).null)
        ImageIOLib.CFRelease(url)
        raise "Failed to create image source for: #{path}" if source.null?

        # Get image
        cg_image = ImageIOLib.CGImageSourceCreateImageAtIndex(source, 0, Pointer(Void).null)
        ImageIOLib.CFRelease(source)
        raise "Failed to load image from: #{path}" if cg_image.null?

        # Get dimensions
        width = ImageIOLib.CGImageGetWidth(cg_image).to_i32
        height = ImageIOLib.CGImageGetHeight(cg_image).to_i32
        channels = 4  # RGBA

        # Create buffer for pixel data
        bytes_per_row = width * channels
        data = Array(UInt8).new(height * bytes_per_row, 0_u8)

        # Create color space
        color_space = ImageIOLib.CGColorSpaceCreateDeviceRGB
        raise "Failed to create color space" if color_space.null?

        # Create bitmap context
        context = ImageIOLib.CGBitmapContextCreate(
          data.to_unsafe.as(Void*),
          width,
          height,
          8,  # bits per component
          bytes_per_row,
          color_space,
          RGBA_BITMAP_INFO
        )
        ImageIOLib.CGColorSpaceRelease(color_space)
        raise "Failed to create bitmap context" if context.null?

        # Draw image into context
        rect = ImageIOLib::CGRect.new(
          x: 0.0,
          y: 0.0,
          width: width.to_f64,
          height: height.to_f64
        )
        ImageIOLib.CGContextDrawImage(context, rect, cg_image)

        # Cleanup
        ImageIOLib.CGContextRelease(context)
        ImageIOLib.CGImageRelease(cg_image)

        Image.new(width, height, channels, data)
      end

      # Load image from bytes
      def self.load_from_bytes(bytes : Bytes) : Image
        # Create CFData
        cf_data = ImageIOLib.CFDataCreate(
          Pointer(Void).null,
          bytes.to_unsafe,
          bytes.size.to_i64
        )
        raise "Failed to create CFData" if cf_data.null?

        # Create image source
        source = ImageIOLib.CGImageSourceCreateWithData(cf_data, Pointer(Void).null)
        ImageIOLib.CFRelease(cf_data)
        raise "Failed to create image source from bytes" if source.null?

        # Get image
        cg_image = ImageIOLib.CGImageSourceCreateImageAtIndex(source, 0, Pointer(Void).null)
        ImageIOLib.CFRelease(source)
        raise "Failed to load image from bytes" if cg_image.null?

        # Get dimensions
        width = ImageIOLib.CGImageGetWidth(cg_image).to_i32
        height = ImageIOLib.CGImageGetHeight(cg_image).to_i32
        channels = 4  # RGBA

        # Create buffer for pixel data
        bytes_per_row = width * channels
        data = Array(UInt8).new(height * bytes_per_row, 0_u8)

        # Create color space
        color_space = ImageIOLib.CGColorSpaceCreateDeviceRGB
        raise "Failed to create color space" if color_space.null?

        # Create bitmap context
        context = ImageIOLib.CGBitmapContextCreate(
          data.to_unsafe.as(Void*),
          width,
          height,
          8,
          bytes_per_row,
          color_space,
          RGBA_BITMAP_INFO
        )
        ImageIOLib.CGColorSpaceRelease(color_space)
        raise "Failed to create bitmap context" if context.null?

        # Draw image into context
        rect = ImageIOLib::CGRect.new(
          x: 0.0,
          y: 0.0,
          width: width.to_f64,
          height: height.to_f64
        )
        ImageIOLib.CGContextDrawImage(context, rect, cg_image)

        # Cleanup
        ImageIOLib.CGContextRelease(context)
        ImageIOLib.CGImageRelease(cg_image)

        Image.new(width, height, channels, data)
      end

      # Load and resize image
      def self.load_resized(path : String, target_width : Int32, target_height : Int32) : Image
        img = load(path)
        resize(img, target_width, target_height)
      end

      # Resize image using bilinear interpolation
      def self.resize(img : Image, new_width : Int32, new_height : Int32) : Image
        return img if img.width == new_width && img.height == new_height

        new_data = Array(UInt8).new(new_height * new_width * img.channels, 0_u8)

        scale_x = img.width.to_f32 / new_width.to_f32
        scale_y = img.height.to_f32 / new_height.to_f32

        new_height.times do |y|
          new_width.times do |x|
            # Source coordinates
            src_x = (x + 0.5_f32) * scale_x - 0.5_f32
            src_y = (y + 0.5_f32) * scale_y - 0.5_f32

            # Clamp to valid range
            src_x = src_x.clamp(0.0_f32, (img.width - 1).to_f32)
            src_y = src_y.clamp(0.0_f32, (img.height - 1).to_f32)

            # Integer and fractional parts
            x0 = src_x.to_i32.clamp(0, img.width - 1)
            y0 = src_y.to_i32.clamp(0, img.height - 1)
            x1 = (x0 + 1).clamp(0, img.width - 1)
            y1 = (y0 + 1).clamp(0, img.height - 1)

            fx = src_x - x0.to_f32
            fy = src_y - y0.to_f32

            img.channels.times do |c|
              # Bilinear interpolation
              v00 = img.data[(y0 * img.width + x0) * img.channels + c].to_f32
              v10 = img.data[(y0 * img.width + x1) * img.channels + c].to_f32
              v01 = img.data[(y1 * img.width + x0) * img.channels + c].to_f32
              v11 = img.data[(y1 * img.width + x1) * img.channels + c].to_f32

              v0 = v00 * (1.0_f32 - fx) + v10 * fx
              v1 = v01 * (1.0_f32 - fx) + v11 * fx
              v = v0 * (1.0_f32 - fy) + v1 * fy

              new_data[(y * new_width + x) * img.channels + c] = v.clamp(0.0_f32, 255.0_f32).to_u8
            end
          end
        end

        Image.new(new_width, new_height, img.channels, new_data)
      end
    end

    # Convenience function
    def self.load_image(path : String) : Image
      ImageIO.load(path)
    end

    def self.load_image_resized(path : String, width : Int32, height : Int32) : Image
      ImageIO.load_resized(path, width, height)
    end
  end
end
