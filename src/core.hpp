#pragma once

#include <vector>
#include <memory>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
}

struct AVFormatContextDeleter {
    void operator()(AVFormatContext *ctx) const {
        if (ctx) avformat_close_input(&ctx);
    }
};

struct AVCodecContextDeleter {
    void operator()(AVCodecContext *ctx) const {
        if (ctx) avcodec_free_context(&ctx);
    }
};

struct AVFormatOutputContextDeleter {
    void operator()(AVFormatContext *ctx) const {
        if (ctx) {
            if (ctx->pb && !(ctx->oformat->flags & AVFMT_NOFILE)) {
                avio_closep(&ctx->pb);
            }
            avformat_free_context(ctx);
        }
    }
};
using UniqueAVFormatOutputContext = std::unique_ptr<AVFormatContext, AVFormatOutputContextDeleter>;

using UniqueAVCodecContext = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;

struct SwrContextDeleter {
    void operator()(SwrContext *ctx) const {
        if (ctx) swr_free(&ctx);
    }
};
using UniqueSwrContext = std::unique_ptr<SwrContext, SwrContextDeleter>;

struct AVPacketDeleter {
    void operator()(AVPacket *pkt) const {
        if (pkt) av_packet_free(&pkt);
    }
};
using UniqueAVPacket = std::unique_ptr<AVPacket, AVPacketDeleter>;

struct AVFrameDeleter {
    void operator()(AVFrame *frame) const {
        if (frame) av_frame_free(&frame);
    }
};
using UniqueAVFrame = std::unique_ptr<AVFrame, AVFrameDeleter>;

using UniqueAVFormatContext = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;

struct UniqueAVChannelLayout {
    AVChannelLayout layout{};
    UniqueAVChannelLayout() {
        av_channel_layout_default(&layout, 2);
    }
    ~UniqueAVChannelLayout() {
        av_channel_layout_uninit(&layout);
    }
    auto get() -> AVChannelLayout * { return &layout; }
};

struct TrtContextOpaque;

auto Trt_Init(const char *modelPath, int *chunkLen, int *numSources, int *maxBatch, bool *isDynamic) -> TrtContextOpaque *;
auto Trt_Process(TrtContextOpaque *hCtx, float *h_input, float *h_output, int batchSize) -> int;
void Trt_Destroy(TrtContextOpaque *hCtx);

void Trt_Sync(TrtContextOpaque *hCtx);
auto Trt_AllocPinned(size_t size) -> void *;
void Trt_FreePinned(void *ptr);

auto GetAudioDuration(const char *path) -> double;