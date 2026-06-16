#include "core.hpp"
#include "AudioStream.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <optional>
#include <array>
#include <cxxopts.hpp>

namespace fs = std::filesystem;

constexpr int   MODEL_SAMPLE_RATE = 44100;
constexpr float OVERLAP_RATIO     = 0.25f;

const std::vector<std::string> Sources6 = { "drums", "bass", "other", "vocals", "guitar", "piano" };
const std::vector<std::string> Sources4 = { "drums", "bass", "other", "vocals" };

struct Config {
    std::string              InputPath;
    std::string              ModelPath;
    std::string              OutputDir = "stems";
    std::string              Format    = "wav";
    std::vector<std::string> SelectedStems;
    int                      BatchSize = 1;
};

struct PinnedDeleter {
    void operator()(float *ptr) const { Trt_FreePinned(ptr); }
};
using PinnedFloatPtr = std::unique_ptr<float[], PinnedDeleter>;

auto RunInferenceStreaming(TrtContextOpaque *ctx, Config *cfg, int chunkLen, int numSources, const std::string &outputDir, double durationSec, bool isDynamic, int maxBatch) -> bool {
    int     overlapFrames    = static_cast<int>(OVERLAP_RATIO * chunkLen);
    int     hopLen           = chunkLen - overlapFrames;
    int64_t totalSamples     = static_cast<int64_t>(durationSec * MODEL_SAMPLE_RATE);
    int64_t processedSamples = 0;
    auto    startTime        = std::chrono::high_resolution_clock::now();

    AudioStreamReader reader(cfg->InputPath, MODEL_SAMPLE_RATE);

    std::vector<std::unique_ptr<AudioStreamWriter>> writers(numSources);
    std::vector<std::string>                        stemNames = (numSources == 6) ? Sources6 : Sources4;
    for (int s = 0; s < numSources; s++) {
        if (!cfg->SelectedStems.empty()) {
            if (std::find(cfg->SelectedStems.begin(), cfg->SelectedStems.end(), stemNames[s]) == cfg->SelectedStems.end()) {
                continue;
            }
        }
        std::string outPath = (fs::path(outputDir) / (stemNames[s] + "." + cfg->Format)).string();
        writers[s]          = std::make_unique<AudioStreamWriter>(outPath, MODEL_SAMPLE_RATE, cfg->Format);
    }

    size_t maxInBytes  = maxBatch * 2 * chunkLen * sizeof(float);
    size_t maxOutBytes = maxBatch * numSources * 2 * chunkLen * sizeof(float);

    std::array<PinnedFloatPtr, 2> inBuf;
    std::array<PinnedFloatPtr, 2> outBuf;
    for (int i = 0; i < 2; i++) {
        inBuf[i].reset(static_cast<float *>(Trt_AllocPinned(maxInBytes)));
        outBuf[i].reset(static_cast<float *>(Trt_AllocPinned(maxOutBytes)));
    }

    std::vector<std::vector<float>> overlapHistL(numSources, std::vector<float>(overlapFrames, 0.0f));
    std::vector<std::vector<float>> overlapHistR(numSources, std::vector<float>(overlapFrames, 0.0f));
    std::vector<std::vector<float>> readyL_pool(numSources, std::vector<float>(chunkLen));
    std::vector<std::vector<float>> readyR_pool(numSources, std::vector<float>(chunkLen));
    std::vector<float>              chunkL(chunkLen, 0.0f);
    std::vector<float>              chunkR(chunkLen, 0.0f);

    struct ChunkMeta {
        int   validSamples    = 0;
        int   samplesAdvanced = 0;
        float mean            = 0.0f;
        float stdDev          = 0.0f;
        int   chunkIndex      = 0;
    };

    struct BatchMeta {
        int                    numChunks = 0;
        std::vector<ChunkMeta> chunks;
        bool                   valid = false;
    } meta[2];
    meta[0].chunks.resize(maxBatch);
    meta[1].chunks.resize(maxBatch);

    bool hasMoreData    = true;
    bool gpuRunning     = false;
    int  batchIdx       = 0;
    int  globalChunkIdx = 0;

    while (hasMoreData || gpuRunning) {
        int activeIdx  = batchIdx % 2;
        int processIdx = (batchIdx + 1) % 2;

        if (hasMoreData) {
            int chunksRead        = 0;
            meta[activeIdx].valid = false;

            for (int b = 0; b < maxBatch; b++) {
                int samplesAdvanced = reader.Read(chunkL, chunkR, chunkLen, hopLen);
                if (samplesAdvanced == 0) {
                    hasMoreData = false;
                    break;
                }

                int validSamples = chunkL.size();
                processedSamples += samplesAdvanced;

                if (chunkL.size() < chunkLen) {
                    chunkL.resize(chunkLen, 0.0f);
                    chunkR.resize(chunkLen, 0.0f);
                }

                double sum = 0, sumSq = 0;
                for (int i = 0; i < chunkLen; i++)
                    sum += (chunkL[i] + chunkR[i]) * 0.5;
                float mean = static_cast<float>(sum / chunkLen);
                for (int i = 0; i < chunkLen; i++) {
                    // sumSq += std::pow((chunkL[i] + chunkR[i]) * 0.5 - mean, 2);
                    float diff = ((chunkL[i] + chunkR[i]) * 0.5f) - mean;
                    sumSq += diff * diff;
                }
                float stdDev = static_cast<float>(std::sqrt(sumSq / chunkLen)) + 1e-8f;

                int inOffset = b * 2 * chunkLen;
                for (int i = 0; i < chunkLen; i++) {
                    inBuf[activeIdx].get()[inOffset + i]            = (chunkL[i] - mean) / stdDev;
                    inBuf[activeIdx].get()[inOffset + chunkLen + i] = (chunkR[i] - mean) / stdDev;
                }

                meta[activeIdx].chunks[b] = { validSamples, samplesAdvanced, mean, stdDev, globalChunkIdx++ };
                chunksRead++;
            }

            meta[activeIdx].numChunks = chunksRead;
            if (chunksRead > 0) meta[activeIdx].valid = true;
        }

        if (gpuRunning) {
            Trt_Sync(ctx);
            gpuRunning = false;

            auto  &m   = meta[processIdx];
            float *out = outBuf[processIdx].get();

            for (int b = 0; b < m.numChunks; b++) {
                auto &cm             = m.chunks[b];
                int   outOffsetBatch = b * numSources * 2 * chunkLen;

                for (int s = 0; s < numSources; s++) {
                    auto &readyL = readyL_pool[s];
                    auto &readyR = readyR_pool[s];

                    // [OPTIMIZATION: REMOVED] Remove readyL.resize() and readyR.resize()!
                    // The capacity is already declared full (chunkLen) at the start of allocation.
                    // readyL.resize(cm.samplesAdvanced);
                    // readyR.resize(cm.samplesAdvanced);

                    int srcOffset = outOffsetBatch + s * 2 * chunkLen;

                    for (int t = 0; t < chunkLen; t++) {
                        float valL = (out[srcOffset + t] * cm.stdDev) + cm.mean;
                        float valR = (out[srcOffset + chunkLen + t] * cm.stdDev) + cm.mean;

                        float w = 1.0f;
                        if (cm.chunkIndex > 0 && t < overlapFrames)
                            w = static_cast<float>(t) / overlapFrames;
                        else if (cm.validSamples == chunkLen && t >= hopLen)
                            w = static_cast<float>(chunkLen - t) / overlapFrames;

                        valL *= w;
                        valR *= w;

                        if (t < overlapFrames) {
                            if (cm.chunkIndex == 0) {
                                if (t < cm.samplesAdvanced) {
                                    readyL[t] = valL;
                                    readyR[t] = valR;
                                }
                            } else {
                                if (t < cm.samplesAdvanced) {
                                    readyL[t] = valL + overlapHistL[s][t];
                                    readyR[t] = valR + overlapHistR[s][t];
                                }
                            }
                        } else if (t < hopLen) {
                            if (t < cm.samplesAdvanced) {
                                readyL[t] = valL;
                                readyR[t] = valR;
                            }
                        } else {
                            overlapHistL[s][t - hopLen] = valL;
                            overlapHistR[s][t - hopLen] = valR;
                        }
                    }
                    if (writers[s]) writers[s]->Write(readyL.data(), readyR.data(), cm.samplesAdvanced);
                }
            }
        }

        if (hasMoreData && meta[activeIdx].valid) {
            int runBatchSize = meta[activeIdx].numChunks;

            if (!isDynamic && runBatchSize < maxBatch) {
                size_t paddingStartOffset = runBatchSize * 2 * chunkLen;
                size_t paddingBytes       = (maxBatch - runBatchSize) * 2 * chunkLen * sizeof(float);
                std::memset(inBuf[activeIdx].get() + paddingStartOffset, 0, paddingBytes);
                runBatchSize = maxBatch;
            }

            int status = Trt_Process(ctx, inBuf[activeIdx].get(), outBuf[activeIdx].get(), runBatchSize);
            if (status != 0) {
                std::cerr << "\n[Error] Trt_Process failed with code: " << status << "\n";
                return false;
            }
            gpuRunning = true;
        }

        if (totalSamples > 0 && (batchIdx % 5 == 0 || !hasMoreData)) {
            int pct = std::min(100, std::max(0, static_cast<int>((static_cast<double>(processedSamples) / totalSamples) * 100.0)));
            std::cout << "\r  + Separating... " << pct << "%" << std::flush;
        }

        batchIdx++;
    }

    for (int s = 0; s < numSources; s++) {
        if (writers[s]) writers[s]->Close();
    }

    auto                          endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    std::cout << "\r  + Separating & Encoding... Done in " << std::fixed << std::setprecision(2) << elapsed.count() << "s   \n";

    return true;
}

auto ParseArgs(int argc, char *argv[], const std::string &baseDir) -> std::optional<Config> {
    cxxopts::Options options("DemucsTRT", "Separating audio using TensorRT");

    options.add_options()("m,model", "[Required] Path to the TensorRT model file", cxxopts::value<std::string>())("input", "[Required] Input audio/video path.", cxxopts::value<std::string>())("o,output", "Output directory path.", cxxopts::value<std::string>()->default_value(fs::path(baseDir).append("stems").string()))("f,format", "Output format audio (e.g., wav, mp3, flac).", cxxopts::value<std::string>()->default_value("wav"))("s,stems", "Specific stems to extract (e.g., vocals, drums).", cxxopts::value<std::vector<std::string>>()->default_value({}))("b,batch", "Batch size for inference", cxxopts::value<int>()->default_value("1"))("h,help", "Print usage instructions.");

    options.parse_positional("input", cxxopts::PositionalMode::Replace);

    options.positional_help("[file]");

    options.allow_unrecognised_options();

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << '\n';
        exit(0);
    }

    if (result.count("model") == 0) {
        std::cout << "Error: Missing required option '--model' (-m)" << '\n';
        std::cout << options.help() << '\n';
        return std::nullopt;
    }

    if (result.count("input") == 0) {
        std::cout << "Error: Missing input file" << '\n';
        std::cout << options.help() << '\n';
        return std::nullopt;
    }

    Config cfg;
    cfg.ModelPath     = result["model"].as<std::string>();
    cfg.InputPath     = result["input"].as<std::string>();
    cfg.OutputDir     = result["output"].as<std::string>();
    cfg.Format        = result["format"].as<std::string>();
    cfg.SelectedStems = result["stems"].as<std::vector<std::string>>();
    cfg.BatchSize     = result["batch"].as<int>();

    return cfg;
}

auto main(int argc, char *argv[]) -> int {
    std::string baseDir = fs::absolute(fs::path(argv[0])).parent_path().string();

    auto cfgOpt = ParseArgs(argc, argv, baseDir);
    if (!cfgOpt) return 1;
    Config &cfg = *cfgOpt;

    if (!fs::exists(cfg.InputPath)) {
        std::cout << "[Error] Input file not found: " << cfg.InputPath << "\n";
        return 1;
    }

    std::string modelPath = fs::absolute(cfg.ModelPath).string();
    if (!fs::exists(modelPath)) {
        std::cout << "[Error] Model file not found: " << modelPath << "\n";
        return 1;
    }

    fs::path outDirBase = fs::path(cfg.OutputDir) / fs::path(cfg.InputPath).stem();
    fs::create_directories(outDirBase);
    std::string outputDir = fs::absolute(outDirBase).string();

    std::cout << std::string(60, '=') << "\n";
    std::cout << "  Input:   " << fs::path(cfg.InputPath).filename().string() << "\n";
    std::cout << "  Model:   " << fs::path(modelPath).filename().string() << "\n";
    std::cout << "  Output:  " << outputDir << "\n";
    std::cout << std::string(60, '=') << "\n";

    std::cout << "  + Probing Audio Metadata... ";

    double durationSec = GetAudioDuration(cfg.InputPath.c_str());

    if (durationSec < 0) {
        std::cout << "UNKNOWN.\n[Warning] Stream mode or missing duration metadata. Progress % won't be calculated.\n";
        durationSec = 0.0;
    } else {
        std::cout << durationSec << " seconds.\n";
    }

    std::cout << durationSec << " seconds.\n";

    using TrtCtxPtr = std::unique_ptr<TrtContextOpaque, decltype(&Trt_Destroy)>;

    std::cout << "  + Initializing TensorRT Engine... ";

    int  chunkLen = 0, numSources = 0, maxBatchLimit = 1;
    bool isDynamic = false;

    TrtCtxPtr ctx(Trt_Init(modelPath.c_str(), &chunkLen, &numSources, &maxBatchLimit, &isDynamic), Trt_Destroy);

    if (!ctx) {
        std::cout << "FAILED.\n";
        std::cout << "  [Hint] Check that nvinfer_*.dll, and cudart64_*.dll\n";
        std::cout << "         are present next to the exe.\n";
        return 1;
    }

    int actualBatch = cfg.BatchSize;
    if (actualBatch > maxBatchLimit) {
        std::cout << "[Warning] Requested batch size (" << actualBatch << ") exceeds model limit. Using " << maxBatchLimit << "\n";
        actualBatch = maxBatchLimit;
    } else if (!isDynamic && actualBatch != maxBatchLimit) {
        std::cout << "[Warning] Static model requires fixed batch. Using " << maxBatchLimit << "\n";
        actualBatch = maxBatchLimit;
    }
    cfg.BatchSize = actualBatch;

    std::cout << "OK\n";
    std::cout << "  + Model Type : " << (isDynamic ? "Dynamic Shape" : "Static Shape") << "\n";
    std::cout << "  + Chunk Size : " << chunkLen << "\n";
    std::cout << "  + Batch Size : " << actualBatch << " (Max: " << maxBatchLimit << ")\n";

    if (!RunInferenceStreaming(ctx.get(), &cfg, chunkLen, numSources, outputDir, durationSec, isDynamic, actualBatch)) {
        return 1;
    }

    return 0;
}