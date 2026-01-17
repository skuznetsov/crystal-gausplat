// FFmpeg struct field accessors for Crystal FFI
// These are needed because FFmpeg structs are complex and we need
// to access specific fields without importing the full struct layout

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

// ============================================================================
// Format context accessors
// ============================================================================

unsigned int gs_av_nb_streams(AVFormatContext *ctx) {
    return ctx->nb_streams;
}

AVStream* gs_av_get_stream(AVFormatContext *ctx, unsigned int index) {
    if (index >= ctx->nb_streams) return NULL;
    return ctx->streams[index];
}

int64_t gs_av_duration(AVFormatContext *ctx) {
    return ctx->duration;
}

// ============================================================================
// Stream accessors
// ============================================================================

AVCodecParameters* gs_av_stream_codecpar(AVStream *stream) {
    return stream->codecpar;
}

AVRational gs_av_stream_time_base(AVStream *stream) {
    return stream->time_base;
}

AVRational gs_av_stream_avg_frame_rate(AVStream *stream) {
    return stream->avg_frame_rate;
}

int64_t gs_av_stream_nb_frames(AVStream *stream) {
    return stream->nb_frames;
}

// ============================================================================
// Codec parameters accessors
// ============================================================================

int gs_av_codecpar_width(AVCodecParameters *par) {
    return par->width;
}

int gs_av_codecpar_height(AVCodecParameters *par) {
    return par->height;
}

int gs_av_codecpar_codec_id(AVCodecParameters *par) {
    return par->codec_id;
}

int gs_av_codecpar_codec_type(AVCodecParameters *par) {
    return par->codec_type;
}

int gs_av_codecpar_format(AVCodecParameters *par) {
    return par->format;
}

// ============================================================================
// Codec context accessors
// ============================================================================

int gs_av_codec_ctx_width(AVCodecContext *ctx) {
    return ctx->width;
}

int gs_av_codec_ctx_height(AVCodecContext *ctx) {
    return ctx->height;
}

int gs_av_codec_ctx_pix_fmt(AVCodecContext *ctx) {
    return ctx->pix_fmt;
}

// ============================================================================
// Frame accessors
// ============================================================================

uint8_t* gs_av_frame_data(AVFrame *frame, int plane) {
    return frame->data[plane];
}

int gs_av_frame_linesize(AVFrame *frame, int plane) {
    return frame->linesize[plane];
}

int gs_av_frame_width(AVFrame *frame) {
    return frame->width;
}

int gs_av_frame_height(AVFrame *frame) {
    return frame->height;
}

int gs_av_frame_format(AVFrame *frame) {
    return frame->format;
}

int64_t gs_av_frame_pts(AVFrame *frame) {
    return frame->pts;
}

void gs_av_frame_set_format(AVFrame *frame, int format) {
    frame->format = format;
}

void gs_av_frame_set_width(AVFrame *frame, int width) {
    frame->width = width;
}

void gs_av_frame_set_height(AVFrame *frame, int height) {
    frame->height = height;
}

// ============================================================================
// Packet accessors
// ============================================================================

int gs_av_packet_stream_index(AVPacket *pkt) {
    return pkt->stream_index;
}

int64_t gs_av_packet_pts(AVPacket *pkt) {
    return pkt->pts;
}

int64_t gs_av_packet_dts(AVPacket *pkt) {
    return pkt->dts;
}
