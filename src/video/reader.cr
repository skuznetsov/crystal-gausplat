# Video file reader using FFmpeg
# Supports MOV, MP4, MKV, AVI, WebM, and more

require "./ffmpeg"
require "../core/tensor"

module GS
  module Video
    # Video metadata
    struct VideoInfo
      property width : Int32
      property height : Int32
      property fps : Float64
      property duration : Float64      # seconds
      property frame_count : Int64
      property codec_name : String

      def initialize(@width, @height, @fps, @duration, @frame_count, @codec_name)
      end
    end

    # A decoded video frame
    struct Frame
      property data : Tensor          # RGB tensor [H, W, 3]
      property timestamp : Float64    # seconds
      property frame_number : Int64

      def initialize(@data, @timestamp, @frame_number)
      end
    end

    # Video file reader
    class Reader
      getter info : VideoInfo
      getter path : String

      @format_ctx : LibAV::AVFormatContext*
      @codec_ctx : LibAV::AVCodecContext*
      @sws_ctx : LibAV::SwsContext*
      @video_stream_index : Int32
      @frame : LibAV::AVFrame*
      @frame_rgb : LibAV::AVFrame*
      @packet : LibAV::AVPacket*
      @time_base : LibAV::AVRationalStruct
      @current_frame : Int64 = 0_i64

      def initialize(@path : String)
        # Suppress FFmpeg logs (set to AV_LOG_ERROR for debugging)
        LibAV.av_log_set_level(LibAV::AV_LOG_QUIET)

        # Open input file
        @format_ctx = Pointer(LibAV::AVFormatContext).null
        format_ctx_ptr = pointerof(@format_ctx)
        ret = LibAV.avformat_open_input(format_ctx_ptr, @path.to_unsafe, nil, nil)
        raise "Failed to open video: #{Video.av_error_string(ret)}" if ret < 0

        # Find stream info
        ret = LibAV.avformat_find_stream_info(@format_ctx, nil)
        raise "Failed to find stream info: #{Video.av_error_string(ret)}" if ret < 0

        # Find video stream
        codec_ptr = Pointer(LibAV::AVCodec).null
        @video_stream_index = LibAV.av_find_best_stream(
          @format_ctx, LibAV::AVMEDIA_TYPE_VIDEO, -1, -1, pointerof(codec_ptr), 0
        )
        raise "No video stream found" if @video_stream_index < 0

        # Get stream
        stream = LibAV.gs_av_get_stream(@format_ctx, @video_stream_index.to_u32)
        raise "Failed to get stream" if stream.null?

        @time_base = LibAV.gs_av_stream_time_base(stream)

        # Get codec parameters
        codecpar = LibAV.gs_av_stream_codecpar(stream)
        codec_id = LibAV.gs_av_codecpar_codec_id(codecpar)

        # Find decoder
        codec = LibAV.avcodec_find_decoder(codec_id)
        raise "Decoder not found for codec #{codec_id}" if codec.null?

        # Allocate codec context
        @codec_ctx = LibAV.avcodec_alloc_context3(codec)
        raise "Failed to allocate codec context" if @codec_ctx.null?

        # Copy parameters to context
        ret = LibAV.avcodec_parameters_to_context(@codec_ctx, codecpar)
        raise "Failed to copy codec parameters: #{Video.av_error_string(ret)}" if ret < 0

        # Open codec
        ret = LibAV.avcodec_open2(@codec_ctx, codec, nil)
        raise "Failed to open codec: #{Video.av_error_string(ret)}" if ret < 0

        # Get video info
        width = LibAV.gs_av_codec_ctx_width(@codec_ctx)
        height = LibAV.gs_av_codec_ctx_height(@codec_ctx)
        pix_fmt = LibAV.gs_av_codec_ctx_pix_fmt(@codec_ctx)

        frame_rate = LibAV.gs_av_stream_avg_frame_rate(stream)
        fps = Video.rational_to_f(frame_rate)
        fps = 30.0 if fps <= 0  # fallback

        duration_ticks = LibAV.gs_av_duration(@format_ctx)
        duration = duration_ticks.to_f64 / 1_000_000.0  # AV_TIME_BASE = 1000000

        frame_count = LibAV.gs_av_stream_nb_frames(stream)
        frame_count = (duration * fps).to_i64 if frame_count <= 0

        @info = VideoInfo.new(
          width: width,
          height: height,
          fps: fps,
          duration: duration,
          frame_count: frame_count,
          codec_name: "video"  # TODO: get actual codec name
        )

        # Allocate frames
        @frame = LibAV.av_frame_alloc
        raise "Failed to allocate frame" if @frame.null?

        @frame_rgb = LibAV.av_frame_alloc
        raise "Failed to allocate RGB frame" if @frame_rgb.null?

        # Setup RGB frame
        LibAV.gs_av_frame_set_format(@frame_rgb, LibAV::AV_PIX_FMT_RGB24)
        LibAV.gs_av_frame_set_width(@frame_rgb, width)
        LibAV.gs_av_frame_set_height(@frame_rgb, height)
        ret = LibAV.av_frame_get_buffer(@frame_rgb, 32)
        raise "Failed to allocate RGB frame buffer: #{Video.av_error_string(ret)}" if ret < 0

        # Create swscale context for pixel format conversion
        @sws_ctx = LibAV.sws_getContext(
          width, height, pix_fmt,
          width, height, LibAV::AV_PIX_FMT_RGB24,
          LibAV::SWS_BILINEAR, nil, nil, nil
        )
        raise "Failed to create swscale context" if @sws_ctx.null?

        # Allocate packet
        @packet = LibAV.av_packet_alloc
        raise "Failed to allocate packet" if @packet.null?
      end

      def finalize
        close
      end

      def close
        return if @format_ctx.null?

        pkt = @packet
        LibAV.av_packet_free(pointerof(pkt)) unless @packet.null?
        @packet = Pointer(LibAV::AVPacket).null

        frame = @frame
        LibAV.av_frame_free(pointerof(frame)) unless @frame.null?
        @frame = Pointer(LibAV::AVFrame).null

        frame_rgb = @frame_rgb
        LibAV.av_frame_free(pointerof(frame_rgb)) unless @frame_rgb.null?
        @frame_rgb = Pointer(LibAV::AVFrame).null

        LibAV.sws_freeContext(@sws_ctx) unless @sws_ctx.null?
        @sws_ctx = Pointer(LibAV::SwsContext).null

        codec = @codec_ctx
        LibAV.avcodec_free_context(pointerof(codec)) unless @codec_ctx.null?
        @codec_ctx = Pointer(LibAV::AVCodecContext).null

        fmt = @format_ctx
        LibAV.avformat_close_input(pointerof(fmt))
        @format_ctx = Pointer(LibAV::AVFormatContext).null
      end

      # Read next frame
      def read_frame : Frame?
        loop do
          ret = LibAV.av_read_frame(@format_ctx, @packet)

          if ret < 0
            # EOF or error
            return nil if ret == LibAV::AVERROR_EOF || ret == -1
            raise "Error reading frame: #{Video.av_error_string(ret)}"
          end

          # Skip non-video packets
          if LibAV.gs_av_packet_stream_index(@packet) != @video_stream_index
            LibAV.av_packet_unref(@packet)
            next
          end

          # Send packet to decoder
          ret = LibAV.avcodec_send_packet(@codec_ctx, @packet)
          LibAV.av_packet_unref(@packet)

          if ret < 0
            next  # Try next packet
          end

          # Receive decoded frame
          ret = LibAV.avcodec_receive_frame(@codec_ctx, @frame)

          if ret >= 0
            # Got a frame, convert to RGB
            frame = convert_frame_to_tensor
            @current_frame += 1
            return frame
          end

          # EAGAIN means we need more packets
          next if ret == -11  # AVERROR(EAGAIN)

          # Other error
          break
        end

        nil
      end

      # Seek to specific time (seconds)
      def seek(time_sec : Float64) : Bool
        # Convert to stream time base
        timestamp = (time_sec * @time_base.den.to_f64 / @time_base.num.to_f64).to_i64

        ret = LibAV.av_seek_frame(@format_ctx, @video_stream_index, timestamp, 0)

        if ret >= 0
          LibAV.avcodec_flush_buffers(@codec_ctx)
          @current_frame = (time_sec * @info.fps).to_i64
          true
        else
          false
        end
      end

      # Seek to specific frame number
      def seek_frame(frame_num : Int64) : Bool
        time_sec = frame_num.to_f64 / @info.fps
        seek(time_sec)
      end

      # Read all frames (use with caution for long videos!)
      def read_all : Array(Frame)
        frames = [] of Frame
        while frame = read_frame
          frames << frame
        end
        frames
      end

      # Read frames at regular intervals
      def read_sampled(interval_sec : Float64) : Array(Frame)
        frames = [] of Frame
        next_time = 0.0

        while next_time < @info.duration
          if seek(next_time)
            if frame = read_frame
              frames << frame
            end
          end
          next_time += interval_sec
        end

        frames
      end

      # Read N frames uniformly distributed across video
      def read_uniform(n_frames : Int32) : Array(Frame)
        return [] of Frame if n_frames <= 0

        interval = @info.duration / n_frames
        read_sampled(interval)
      end

      # Iterate over frames with block
      def each_frame(&block : Frame -> _)
        while frame = read_frame
          block.call(frame)
        end
      end

      # Iterate over sampled frames
      def each_sampled(interval_sec : Float64, &block : Frame -> _)
        next_time = 0.0

        while next_time < @info.duration
          if seek(next_time)
            if frame = read_frame
              block.call(frame)
            end
          end
          next_time += interval_sec
        end
      end

      private def convert_frame_to_tensor : Frame
        width = @info.width
        height = @info.height

        # Get source data pointers
        src_data = StaticArray(Pointer(UInt8), 4).new(Pointer(UInt8).null)
        src_linesize = StaticArray(Int32, 4).new(0)

        4.times do |i|
          src_data[i] = LibAV.gs_av_frame_data(@frame, i)
          src_linesize[i] = LibAV.gs_av_frame_linesize(@frame, i)
        end

        # Get destination pointers
        dst_data = StaticArray(Pointer(UInt8), 4).new(Pointer(UInt8).null)
        dst_linesize = StaticArray(Int32, 4).new(0)

        4.times do |i|
          dst_data[i] = LibAV.gs_av_frame_data(@frame_rgb, i)
          dst_linesize[i] = LibAV.gs_av_frame_linesize(@frame_rgb, i)
        end

        # Convert pixel format
        LibAV.sws_scale(
          @sws_ctx,
          src_data.to_unsafe, src_linesize.to_unsafe,
          0, height,
          dst_data.to_unsafe, dst_linesize.to_unsafe
        )

        # Copy RGB data to tensor
        tensor = Tensor.new(height, width, 3, device: Tensor::Device::CPU)
        tensor_data = tensor.cpu_data.not_nil!

        rgb_data = dst_data[0]
        rgb_linesize = dst_linesize[0]

        height.times do |y|
          row_ptr = rgb_data + y * rgb_linesize
          width.times do |x|
            r = row_ptr[x * 3 + 0]
            g = row_ptr[x * 3 + 1]
            b = row_ptr[x * 3 + 2]

            # Store as float [0, 1]
            idx = (y * width + x) * 3
            tensor_data[idx + 0] = r.to_f32 / 255.0_f32
            tensor_data[idx + 1] = g.to_f32 / 255.0_f32
            tensor_data[idx + 2] = b.to_f32 / 255.0_f32
          end
        end

        # Calculate timestamp
        pts = LibAV.gs_av_frame_pts(@frame)
        timestamp = pts.to_f64 * @time_base.num.to_f64 / @time_base.den.to_f64

        Frame.new(tensor, timestamp, @current_frame)
      end
    end
  end
end
