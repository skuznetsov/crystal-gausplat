# FFmpeg FFI bindings
# Links against libavformat, libavcodec, libavutil, libswscale

@[Link("avformat")]
@[Link("avcodec")]
@[Link("avutil")]
@[Link("swscale")]
lib LibAV
  # ============================================================================
  # Constants
  # ============================================================================

  AVMEDIA_TYPE_VIDEO = 0
  AVMEDIA_TYPE_AUDIO = 1

  AV_PIX_FMT_RGB24 = 2
  AV_PIX_FMT_RGBA  = 26
  AV_PIX_FMT_BGR24 = 3

  AVERROR_EOF = -541478725  # AVERROR_EOF value

  AV_LOG_QUIET   = -8
  AV_LOG_PANIC   =  0
  AV_LOG_FATAL   =  8
  AV_LOG_ERROR   = 16
  AV_LOG_WARNING = 24
  AV_LOG_INFO    = 32
  AV_LOG_VERBOSE = 40
  AV_LOG_DEBUG   = 48

  SWS_BILINEAR  = 2
  SWS_BICUBIC   = 4
  SWS_LANCZOS   = 0x200

  # ============================================================================
  # Opaque structs (we only need pointers)
  # ============================================================================

  type AVFormatContext = Void
  type AVCodecContext = Void
  type AVCodec = Void
  type AVFrame = Void
  type AVPacket = Void
  type SwsContext = Void
  type AVCodecParameters = Void
  type AVStream = Void
  type AVDictionary = Void
  type AVRational = Void

  # ============================================================================
  # Structs we need to access fields
  # ============================================================================

  struct AVRationalStruct
    num : Int32
    den : Int32
  end

  # ============================================================================
  # Format context functions
  # ============================================================================

  # Open input file
  fun avformat_open_input(
    ps : AVFormatContext**,
    url : LibC::Char*,
    fmt : Void*,
    options : AVDictionary**
  ) : Int32

  # Find stream info
  fun avformat_find_stream_info(
    ic : AVFormatContext*,
    options : AVDictionary**
  ) : Int32

  # Find best stream of given type
  fun av_find_best_stream(
    ic : AVFormatContext*,
    type : Int32,
    wanted_stream : Int32,
    related_stream : Int32,
    decoder_ret : AVCodec**,
    flags : Int32
  ) : Int32

  # Close input
  fun avformat_close_input(s : AVFormatContext**) : Void

  # Read frame
  fun av_read_frame(s : AVFormatContext*, pkt : AVPacket*) : Int32

  # Seek to timestamp
  fun av_seek_frame(
    s : AVFormatContext*,
    stream_index : Int32,
    timestamp : Int64,
    flags : Int32
  ) : Int32

  # ============================================================================
  # Format context field accessors (we need these because struct is opaque)
  # ============================================================================

  # Get number of streams
  fun gs_av_nb_streams(ctx : AVFormatContext*) : UInt32

  # Get stream at index
  fun gs_av_get_stream(ctx : AVFormatContext*, index : UInt32) : AVStream*

  # Get duration in AV_TIME_BASE units
  fun gs_av_duration(ctx : AVFormatContext*) : Int64

  # ============================================================================
  # Stream accessors
  # ============================================================================

  fun gs_av_stream_codecpar(stream : AVStream*) : AVCodecParameters*
  fun gs_av_stream_time_base(stream : AVStream*) : AVRationalStruct
  fun gs_av_stream_avg_frame_rate(stream : AVStream*) : AVRationalStruct
  fun gs_av_stream_nb_frames(stream : AVStream*) : Int64

  # ============================================================================
  # Codec parameters accessors
  # ============================================================================

  fun gs_av_codecpar_width(par : AVCodecParameters*) : Int32
  fun gs_av_codecpar_height(par : AVCodecParameters*) : Int32
  fun gs_av_codecpar_codec_id(par : AVCodecParameters*) : Int32
  fun gs_av_codecpar_codec_type(par : AVCodecParameters*) : Int32
  fun gs_av_codecpar_format(par : AVCodecParameters*) : Int32

  # ============================================================================
  # Codec functions
  # ============================================================================

  # Find decoder by ID
  fun avcodec_find_decoder(id : Int32) : AVCodec*

  # Allocate codec context
  fun avcodec_alloc_context3(codec : AVCodec*) : AVCodecContext*

  # Copy parameters to context
  fun avcodec_parameters_to_context(
    codec : AVCodecContext*,
    par : AVCodecParameters*
  ) : Int32

  # Open codec
  fun avcodec_open2(
    avctx : AVCodecContext*,
    codec : AVCodec*,
    options : AVDictionary**
  ) : Int32

  # Free codec context
  fun avcodec_free_context(avctx : AVCodecContext**) : Void

  # Send packet to decoder
  fun avcodec_send_packet(avctx : AVCodecContext*, avpkt : AVPacket*) : Int32

  # Receive decoded frame
  fun avcodec_receive_frame(avctx : AVCodecContext*, frame : AVFrame*) : Int32

  # Flush buffers
  fun avcodec_flush_buffers(avctx : AVCodecContext*) : Void

  # ============================================================================
  # Codec context accessors
  # ============================================================================

  fun gs_av_codec_ctx_width(ctx : AVCodecContext*) : Int32
  fun gs_av_codec_ctx_height(ctx : AVCodecContext*) : Int32
  fun gs_av_codec_ctx_pix_fmt(ctx : AVCodecContext*) : Int32

  # ============================================================================
  # Frame functions
  # ============================================================================

  # Allocate frame
  fun av_frame_alloc : AVFrame*

  # Free frame
  fun av_frame_free(frame : AVFrame**) : Void

  # Get frame buffer
  fun av_frame_get_buffer(frame : AVFrame*, align : Int32) : Int32

  # Unref frame
  fun av_frame_unref(frame : AVFrame*) : Void

  # ============================================================================
  # Frame field accessors
  # ============================================================================

  fun gs_av_frame_data(frame : AVFrame*, plane : Int32) : UInt8*
  fun gs_av_frame_linesize(frame : AVFrame*, plane : Int32) : Int32
  fun gs_av_frame_width(frame : AVFrame*) : Int32
  fun gs_av_frame_height(frame : AVFrame*) : Int32
  fun gs_av_frame_format(frame : AVFrame*) : Int32
  fun gs_av_frame_pts(frame : AVFrame*) : Int64
  fun gs_av_frame_set_format(frame : AVFrame*, format : Int32) : Void
  fun gs_av_frame_set_width(frame : AVFrame*, width : Int32) : Void
  fun gs_av_frame_set_height(frame : AVFrame*, height : Int32) : Void

  # ============================================================================
  # Packet functions
  # ============================================================================

  # Allocate packet
  fun av_packet_alloc : AVPacket*

  # Free packet
  fun av_packet_free(pkt : AVPacket**) : Void

  # Unref packet
  fun av_packet_unref(pkt : AVPacket*) : Void

  # ============================================================================
  # Packet accessors
  # ============================================================================

  fun gs_av_packet_stream_index(pkt : AVPacket*) : Int32
  fun gs_av_packet_pts(pkt : AVPacket*) : Int64
  fun gs_av_packet_dts(pkt : AVPacket*) : Int64

  # ============================================================================
  # Swscale functions
  # ============================================================================

  # Get swscale context
  fun sws_getContext(
    srcW : Int32, srcH : Int32, srcFormat : Int32,
    dstW : Int32, dstH : Int32, dstFormat : Int32,
    flags : Int32,
    srcFilter : Void*, dstFilter : Void*, param : Float64*
  ) : SwsContext*

  # Scale/convert frame
  fun sws_scale(
    c : SwsContext*,
    srcSlice : UInt8**, srcStride : Int32*,
    srcSliceY : Int32, srcSliceH : Int32,
    dst : UInt8**, dstStride : Int32*
  ) : Int32

  # Free swscale context
  fun sws_freeContext(swsContext : SwsContext*) : Void

  # ============================================================================
  # Utility functions
  # ============================================================================

  # Set log level
  fun av_log_set_level(level : Int32) : Void

  # Error to string
  fun av_strerror(errnum : Int32, errbuf : LibC::Char*, errbuf_size : LibC::SizeT) : Int32

  # Rescale timestamp
  fun av_rescale_q(a : Int64, bq : AVRationalStruct, cq : AVRationalStruct) : Int64

  # Free pointer
  fun av_free(ptr : Void*) : Void
  fun av_freep(ptr : Void**) : Void

  # Allocate buffer
  fun av_malloc(size : LibC::SizeT) : Void*

  # Image functions
  fun av_image_get_buffer_size(
    pix_fmt : Int32,
    width : Int32,
    height : Int32,
    align : Int32
  ) : Int32

  fun av_image_fill_arrays(
    dst_data : UInt8**,
    dst_linesize : Int32*,
    src : UInt8*,
    pix_fmt : Int32,
    width : Int32,
    height : Int32,
    align : Int32
  ) : Int32
end

module GS
  module Video
    # Helper to convert AV error code to string
    def self.av_error_string(errnum : Int32) : String
      buf = Bytes.new(256)
      LibAV.av_strerror(errnum, buf.to_unsafe.as(LibC::Char*), buf.size)
      String.new(buf.to_unsafe.as(LibC::Char*))
    end

    # Rational to float
    def self.rational_to_f(r : LibAV::AVRationalStruct) : Float64
      return 0.0 if r.den == 0
      r.num.to_f64 / r.den.to_f64
    end
  end
end
