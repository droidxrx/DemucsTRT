#include "AudioStream.hpp"
#include <iostream>
#include <algorithm>

AudioStreamReader::AudioStreamReader(const std::string &path, int targetSr)
    : path_(path), targetSr_(targetSr), eof_(false), audioStreamIdx_(-1) {

    AVFormatContext *raw_fmt_ctx = nullptr;
    if (avformat_open_input(&raw_fmt_ctx, path.c_str(), nullptr, nullptr) != 0) {
        throw std::runtime_error("Reader: Failed to open input " + path);
    }
    formatCtxSafe_ = UniqueAVFormatContext(raw_fmt_ctx);
    avformat_find_stream_info(formatCtxSafe_.get(), nullptr);

    const AVCodec *codec = nullptr;
    audioStreamIdx_      = av_find_best_stream(formatCtxSafe_.get(), AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (audioStreamIdx_ < 0) throw std::runtime_error("Reader: Stream audio not found in " + path);

    codecCtx_ = UniqueAVCodecContext(avcodec_alloc_context3(codec));
    if (avcodec_parameters_to_context(codecCtx_.get(), formatCtxSafe_->streams[audioStreamIdx_]->codecpar) < 0) {
        throw std::runtime_error("Reader: Failed to copy codec parameters to context");
    }

    if (avcodec_open2(codecCtx_.get(), codec, nullptr) < 0) {
        throw std::runtime_error("Reader: Failed to open codec configuration");
    }

    SwrContext *raw_swr_ctx = nullptr;
    if (swr_alloc_set_opts2(&raw_swr_ctx, outChLayout_.get(), AV_SAMPLE_FMT_FLTP, targetSr_, &codecCtx_->ch_layout, codecCtx_->sample_fmt, codecCtx_->sample_rate, 0, nullptr) < 0) {
        throw std::runtime_error("Reader: Failed to allocate SwrContext");
    }

    swrCtx_ = UniqueSwrContext(raw_swr_ctx);

    if (swr_init(swrCtx_.get()) < 0) {
        throw std::runtime_error("Reader: Failed to initialize resampler context");
    }

    packet_   = UniqueAVPacket(av_packet_alloc());
    frame_    = UniqueAVFrame(av_frame_alloc());
    outFrame_ = UniqueAVFrame(av_frame_alloc());
}

AudioStreamReader::~AudioStreamReader() {
}

[[nodiscard]] auto AudioStreamReader::DecodeNextPacket() -> bool {
    bool got_frame = false;

    while (av_read_frame(formatCtxSafe_.get(), packet_.get()) >= 0) {
        if (packet_->stream_index == audioStreamIdx_) {
            if (avcodec_send_packet(codecCtx_.get(), packet_.get()) == 0) {
                while (avcodec_receive_frame(codecCtx_.get(), frame_.get()) == 0) {
                    int out_samples = av_rescale_rnd(
                        swr_get_delay(swrCtx_.get(), codecCtx_->sample_rate) + frame_->nb_samples,
                        targetSr_, codecCtx_->sample_rate, AV_ROUND_UP);

                    outFrame_->ch_layout   = *outChLayout_.get();
                    outFrame_->format      = AV_SAMPLE_FMT_FLTP;
                    outFrame_->sample_rate = targetSr_;
                    outFrame_->nb_samples  = out_samples;

                    av_frame_get_buffer(outFrame_.get(), 0);
                    int ret = swr_convert(swrCtx_.get(), outFrame_->data, outFrame_->nb_samples, (const uint8_t **)frame_->data, frame_->nb_samples);

                    if (ret > 0) {
                        auto *l_ptr = reinterpret_cast<float *>(outFrame_->data[0]);
                        auto *r_ptr = reinterpret_cast<float *>(outFrame_->data[1]);
                        bufferL_.Push(l_ptr, ret);
                        bufferR_.Push(r_ptr, ret);
                    }
                    av_frame_unref(outFrame_.get());
                    got_frame = true;
                }
            }
        }
        av_packet_unref(packet_.get());

        if (got_frame) return true;
    }
    eof_ = true;
    return false;
}

auto AudioStreamReader::Read(std::vector<float> &outL, std::vector<float> &outR, int chunkLen, int hopLen) -> int {
    outL.clear();
    outR.clear();

    while (bufferL_.Size() < chunkLen && !eof_) {
        // Need Handle if return false
        DecodeNextPacket();
    }

    int samplesAvailable = (int)bufferL_.Size();
    if (samplesAvailable == 0) return 0;

    int samplesToTake = std::min(chunkLen, samplesAvailable);

    outL.resize(samplesToTake);
    outR.resize(samplesToTake);
    bufferL_.Read(outL.data(), samplesToTake);
    bufferR_.Read(outR.data(), samplesToTake);

    int samplesAdvanced = std::min(hopLen, samplesAvailable);

    bufferL_.Consume(samplesAdvanced);
    bufferR_.Consume(samplesAdvanced);

    return samplesAdvanced;
}

AudioStreamWriter::AudioStreamWriter(const std::string &path, int targetSr, const std::string &format)
    : path_(path), targetSr_(targetSr), ptsOffset_(0) {

    AVFormatContext *rawFormatCtx = nullptr;
    avformat_alloc_output_context2(&rawFormatCtx, nullptr, nullptr, path.c_str());
    if (!rawFormatCtx) throw std::runtime_error("Writer: Format not supported " + path);
    formatCtx_ = UniqueAVFormatOutputContext(rawFormatCtx);

    const AVCodec *codec = avcodec_find_encoder(formatCtx_->oformat->audio_codec);
    if (!codec) throw std::runtime_error("Writer: Encoder not found");

    stream_ = avformat_new_stream(formatCtx_.get(), codec);

    codecCtx_              = UniqueAVCodecContext(avcodec_alloc_context3(codec));
    codecCtx_->sample_rate = targetSr_;
    codecCtx_->time_base   = { 1, targetSr_ };
    av_channel_layout_default(&codecCtx_->ch_layout, 2);
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 19, 100)
    const void *supported_sample_fmts = nullptr;
    if (avcodec_get_supported_config(codecCtx_.get(), codec, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, &supported_sample_fmts, nullptr) >= 0 && supported_sample_fmts != nullptr) {
        codecCtx_->sample_fmt = *(const enum AVSampleFormat *)supported_sample_fmts;
    } else {
        codecCtx_->sample_fmt = AV_SAMPLE_FMT_FLTP;
    }
#else
    codecCtx_->sample_fmt = codec->sample_fmts ? codec->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
#endif

    if (formatCtx_->oformat->audio_codec != AV_CODEC_ID_PCM_F32LE &&
        formatCtx_->oformat->audio_codec != AV_CODEC_ID_PCM_S16LE) {
        codecCtx_->bit_rate = 192000;
    }
    if (formatCtx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codecCtx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (avcodec_open2(codecCtx_.get(), codec, nullptr) < 0) {
        throw std::runtime_error("Writer: Failed to open encoder codec");
    }

    avcodec_parameters_from_context(stream_->codecpar, codecCtx_.get());

    if (!(formatCtx_->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&formatCtx_->pb, path.c_str(), AVIO_FLAG_WRITE) < 0) {
            throw std::runtime_error("Writer: Failed to open output file " + path);
        }
    }

    if (avformat_write_header(formatCtx_.get(), nullptr) < 0) {
        throw std::runtime_error("Writer: Failed to write format header");
    }

    if (codecCtx_->sample_fmt != AV_SAMPLE_FMT_FLTP) {
        SwrContext *raw_swr_ctx = nullptr;
        swr_alloc_set_opts2(&raw_swr_ctx, &codecCtx_->ch_layout, codecCtx_->sample_fmt, targetSr_, &codecCtx_->ch_layout, AV_SAMPLE_FMT_FLTP, targetSr_, 0, nullptr);
        swrCtx_ = UniqueSwrContext(raw_swr_ctx);
        swr_init(swrCtx_.get());
    }

    encFrame_ = UniqueAVFrame(av_frame_alloc());
    packet_   = UniqueAVPacket(av_packet_alloc());

    encFrame_->format      = codecCtx_->sample_fmt;
    encFrame_->sample_rate = codecCtx_->sample_rate;
    av_channel_layout_copy(&encFrame_->ch_layout, &codecCtx_->ch_layout);

    workerThread_ = std::thread(&AudioStreamWriter::WorkerLoop, this);
}

AudioStreamWriter::~AudioStreamWriter() {
    Close();
}

void AudioStreamWriter::EncodeAndWrite(AVFrame *f) {
    int ret = avcodec_send_frame(codecCtx_.get(), f);
    while (ret >= 0) {
        ret = avcodec_receive_packet(codecCtx_.get(), packet_.get());
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;

        av_packet_rescale_ts(packet_.get(), codecCtx_->time_base, stream_->time_base);
        packet_->stream_index = stream_->index;

        av_interleaved_write_frame(formatCtx_.get(), packet_.get());
        av_packet_unref(packet_.get());
    }
}

void AudioStreamWriter::ProcessInternalBuffer(bool flush) {
    int frameSize = codecCtx_->frame_size > 0 ? codecCtx_->frame_size : 4096;

    while (bufferL_.Size() >= frameSize || (flush && bufferL_.Size() > 0)) {
        int samplesToCopy = std::min(frameSize, static_cast<int>(bufferL_.Size()));

        encFrame_->format      = codecCtx_->sample_fmt;
        encFrame_->sample_rate = codecCtx_->sample_rate;
        av_channel_layout_copy(&encFrame_->ch_layout, &codecCtx_->ch_layout);
        encFrame_->nb_samples = samplesToCopy;

        if (av_frame_get_buffer(encFrame_.get(), 0) < 0) {
            std::cerr << "Writer: Error failed to allocate encoder frame buffer!\n";
            break;
        }

        if (workBufL_.size() < samplesToCopy) {
            workBufL_.resize(samplesToCopy);
            workBufR_.resize(samplesToCopy);
        }

        bufferL_.Read(workBufL_.data(), samplesToCopy);
        bufferR_.Read(workBufR_.data(), samplesToCopy);

        const uint8_t *srcData[2] = {
            reinterpret_cast<const uint8_t *>(workBufL_.data()),
            reinterpret_cast<const uint8_t *>(workBufR_.data())
        };

        if (swrCtx_) {
            swr_convert(swrCtx_.get(), encFrame_->data, samplesToCopy, srcData, samplesToCopy);
        } else {
            memcpy(encFrame_->data[0], srcData[0], samplesToCopy * sizeof(float));
            memcpy(encFrame_->data[1], srcData[1], samplesToCopy * sizeof(float));
        }

        encFrame_->pts = ptsOffset_;
        ptsOffset_ += samplesToCopy;

        EncodeAndWrite(encFrame_.get());
        av_frame_unref(encFrame_.get());

        bufferL_.Consume(samplesToCopy);
        bufferR_.Consume(samplesToCopy);
    }
}

void AudioStreamWriter::Write(const float *L_data, const float *R_data, size_t length) {
    if (length == 0 || formatCtx_ == nullptr) return;

    AudioChunk chunk;
    if (!chunk_pool_.try_dequeue(chunk)) {
        // If the pool is empty, let the vector allocate new memory
        // (This only happens at the beginning of the program)
    }

    // Assign data (std::vector::assign reuses memory if capacity is sufficient)
    chunk.L.assign(L_data, L_data + length);
    chunk.R.assign(R_data, R_data + length);

    // Pure yield surrenders the remainder of execution without forcing the OS to go into a 'sleep' state,
    // ensuring a more reactive (low latency) response time.
    while (queue_size_.load(std::memory_order_relaxed) >= 10 && !stopFlag_) {
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::yield();
    }

    if (stopFlag_.load(std::memory_order_acquire)) return;

    // Enqueue without lock
    queue_.enqueue(std::move(chunk));
    queue_size_.fetch_add(1, std::memory_order_release);
}

void AudioStreamWriter::WorkerLoop() {
    AudioChunk chunk;

    while (!stopFlag_) {
        // Waits for a maximum of 10ms for an item.
        // Uses a timeout to periodically check the stopFlag_.
        if (queue_.wait_dequeue_timed(chunk, std::chrono::milliseconds(10))) {

            queue_size_.fetch_sub(1, std::memory_order_acquire);

            bufferL_.Push(chunk.L.data(), chunk.L.size());
            bufferR_.Push(chunk.R.data(), chunk.R.size());
            ProcessInternalBuffer(false);

            // [OPTIMIZATION] Return to object pool after use
            chunk_pool_.enqueue(std::move(chunk));
        }
    }

    // [IMPORTANT] Drain remaining queue if stopFlag_ is active at program shutdown
    while (queue_.try_dequeue(chunk)) {
        bufferL_.Push(chunk.L.data(), chunk.L.size());
        bufferR_.Push(chunk.R.data(), chunk.R.size());
        ProcessInternalBuffer(false);
    }
}

void AudioStreamWriter::Close() {
    if (!formatCtx_) return;

    stopFlag_ = true; // Directly change flags without needing mutex

    if (workerThread_.joinable()) {
        workerThread_.join();
    }

    ProcessInternalBuffer(true);
    EncodeAndWrite(nullptr);
    av_write_trailer(formatCtx_.get());

    formatCtx_.reset();
}