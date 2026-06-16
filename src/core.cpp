#include "core.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char *msg) noexcept override {

        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << '\n';
    }
} gLogger;

struct TrtContextOpaque {
    std::unique_ptr<IRuntime>          runtime;
    std::unique_ptr<ICudaEngine>       engine;
    std::unique_ptr<IExecutionContext> context;
    void                              *d_input    = nullptr;
    void                              *d_output   = nullptr;
    cudaStream_t                       stream     = nullptr;
    int                                chunkLen   = 0;
    int                                numSources = 0;

    TrtContextOpaque()                                    = default;
    TrtContextOpaque(const TrtContextOpaque &)            = delete;
    TrtContextOpaque &operator=(const TrtContextOpaque &) = delete;
    TrtContextOpaque(TrtContextOpaque &&)                 = delete;
    TrtContextOpaque &operator=(TrtContextOpaque &&)      = delete;

    ~TrtContextOpaque() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (stream) cudaStreamDestroy(stream);
    }
};

auto Trt_Init(const char *modelPath, int *chunkLen, int *numSources, int *maxBatch, bool *isDynamic) -> TrtContextOpaque * {
    auto ctx = std::make_unique<TrtContextOpaque>();

    ctx->runtime.reset(createInferRuntime(gLogger));
    if (!ctx->runtime) {
        return nullptr;
    }

    std::ifstream f(modelPath, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cout << "[TRT] ERROR: Could not open model file: " << modelPath << '\n';
        return nullptr;
    }

    std::streamsize size = f.tellg();
    if (size <= 0) {
        std::cout << "[TRT] ERROR: Model file is empty or invalid: " << modelPath << '\n';
        return nullptr;
    }

    f.seekg(0, std::ios::beg);

    std::vector<char> modelData(static_cast<size_t>(size));
    if (!f.read(modelData.data(), size)) {
        std::cout << "[TRT] ERROR: Failed read model file: " << modelPath << '\n';
        return nullptr;
    }

    ctx->engine.reset(ctx->runtime->deserializeCudaEngine(modelData.data(), size));
    if (!ctx->engine) {
        std::cout << "[TRT] ERROR: Failed to deserialize engine. Wrong TRT version?" << '\n';
        return nullptr;
    }

    ctx->context.reset(ctx->engine->createExecutionContext());
    if (!ctx->context) {
        return nullptr;
    }

    auto inDims  = ctx->engine->getTensorShape("input");
    auto outDims = ctx->engine->getTensorShape("output");

    *isDynamic = (inDims.d[0] == -1);

    if (*isDynamic) {
        // Ambil batas maksimal dari profil ke-0 (OptProfileSelector::kMAX)
        auto maxShape = ctx->engine->getProfileShape("input", 0, nvinfer1::OptProfileSelector::kMAX);
        *maxBatch     = maxShape.d[0];
        ctx->chunkLen = maxShape.d[2];
    } else {
        *maxBatch     = inDims.d[0];
        ctx->chunkLen = inDims.d[2];
    }

    ctx->numSources = (int)outDims.d[1];
    *chunkLen       = ctx->chunkLen;
    *numSources     = ctx->numSources;

    size_t maxInSize  = (*maxBatch) * 2 * ctx->chunkLen * sizeof(float);
    size_t maxOutSize = (*maxBatch) * ctx->numSources * 2 * ctx->chunkLen * sizeof(float);

    if (cudaMalloc(&ctx->d_input, maxInSize) != cudaSuccess ||
        cudaMalloc(&ctx->d_output, maxOutSize) != cudaSuccess) {
        std::cerr << "[TRT] ERROR: CUDA Malloc failed. Out of VRAM!" << '\n';
        return nullptr;
    }

    if (cudaStreamCreate(&ctx->stream) != cudaSuccess) return nullptr;

    ctx->context->setTensorAddress("input", ctx->d_input);
    ctx->context->setTensorAddress("output", ctx->d_output);

    return ctx.release();
}

auto Trt_Process(TrtContextOpaque *hCtx, float *h_input, float *h_output, int batchSize) -> int {
    auto *ctx = hCtx;
    if (!ctx) return -1;

    nvinfer1::Dims dims;
    dims.nbDims = 3;
    dims.d[0]   = batchSize;
    dims.d[1]   = 2;
    dims.d[2]   = ctx->chunkLen;
    ctx->context->setInputShape("input", dims);

    size_t inSize  = batchSize * 2 * ctx->chunkLen * sizeof(float);
    size_t outSize = batchSize * ctx->numSources * 2 * ctx->chunkLen * sizeof(float);

    if (cudaMemcpyAsync(ctx->d_input, h_input, inSize, cudaMemcpyHostToDevice, ctx->stream) != 0) return 1;
    if (!ctx->context->enqueueV3(ctx->stream)) return 2;
    if (cudaMemcpyAsync(h_output, ctx->d_output, outSize, cudaMemcpyDeviceToHost, ctx->stream) != 0) return 3;

    return 0;
}

void Trt_Destroy(TrtContextOpaque *hCtx) {
    delete hCtx;
}

void Trt_Sync(TrtContextOpaque *hCtx) {
    if (hCtx && hCtx->stream) cudaStreamSynchronize(hCtx->stream);
}

auto Trt_AllocPinned(size_t size) -> void * {
    void *ptr = nullptr;
    if (cudaMallocHost(&ptr, size) != cudaSuccess) {
        throw std::bad_alloc();
    }
    return ptr;
}

void Trt_FreePinned(void *ptr) {
    if (ptr) cudaFreeHost(ptr);
}

auto GetAudioDuration(const char *path) -> double {
    AVFormatContext *rawFmtCtx = nullptr;
    if (avformat_open_input(&rawFmtCtx, path, nullptr, nullptr) < 0) {
        return -1.0;
    }

    UniqueAVFormatContext fmtCtx(rawFmtCtx);

    if (avformat_find_stream_info(fmtCtx.get(), nullptr) >= 0) {
        if (fmtCtx->duration != AV_NOPTS_VALUE) {
            return static_cast<double>(fmtCtx->duration) / AV_TIME_BASE;
        }
    }

    return -1.0;
}
