#pragma once

#include "core.hpp"

#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>

#include <concurrentqueue.h>
#include <blockingconcurrentqueue.h>

class AudioRingBuffer {
public:
    explicit AudioRingBuffer(size_t initial_capacity = 8192) {
        buffer_.resize(initial_capacity);
        head_ = 0;
        tail_ = 0;
        size_ = 0;
    }

    void Push(const float *data, size_t len) {
        if (size_ + len > buffer_.size()) {
            Resize(std::max(buffer_.size() * 2, size_ + len));
        }

        size_t right_space = buffer_.size() - tail_;
        if (len <= right_space) {
            std::memcpy(buffer_.data() + tail_, data, len * sizeof(float));
        } else {
            std::memcpy(buffer_.data() + tail_, data, right_space * sizeof(float));
            std::memcpy(buffer_.data(), data + right_space, (len - right_space) * sizeof(float));
        }
        tail_ = (tail_ + len) % buffer_.size();
        size_ += len;
    }

    void Read(float *dest, size_t len) const {
        if (len > size_) len = size_;
        if (len == 0) return;

        size_t right_part = buffer_.size() - head_;
        if (len <= right_part) {
            std::memcpy(dest, buffer_.data() + head_, len * sizeof(float));
        } else {
            std::memcpy(dest, buffer_.data() + head_, right_part * sizeof(float));
            std::memcpy(dest + right_part, buffer_.data(), (len - right_part) * sizeof(float));
        }
    }

    void Consume(size_t len) {
        if (len > size_) len = size_;
        head_ = (head_ + len) % buffer_.size();
        size_ -= len;
    }

    [[nodiscard]] auto Size() const -> size_t { return size_; }
    [[nodiscard]] auto Empty() const -> bool { return size_ == 0; }

private:
    void Resize(size_t new_capacity) {
        std::vector<float> new_buf(new_capacity);
        if (size_ > 0) {
            size_t right_part = buffer_.size() - head_;
            if (size_ <= right_part) {
                std::memcpy(new_buf.data(), buffer_.data() + head_, size_ * sizeof(float));
            } else {
                std::memcpy(new_buf.data(), buffer_.data() + head_, right_part * sizeof(float));
                std::memcpy(new_buf.data() + right_part, buffer_.data(), (size_ - right_part) * sizeof(float));
            }
        }
        buffer_ = std::move(new_buf);
        head_   = 0;
        tail_   = size_ % buffer_.size();
    }

    std::vector<float> buffer_;
    size_t             head_;
    size_t             tail_;
    size_t             size_;
};

class AudioStreamReader {
public:
    AudioStreamReader(const std::string &path, int targetSr);
    ~AudioStreamReader();

    auto Read(std::vector<float> &outL, std::vector<float> &outR, int chunkLen, int hopLen) -> int;

private:
    std::string path_;
    int         targetSr_;
    bool        eof_;
    int         audioStreamIdx_;

    UniqueAVFormatContext formatCtxSafe_ = nullptr;
    UniqueAVCodecContext  codecCtx_      = nullptr;
    UniqueSwrContext      swrCtx_        = nullptr;
    UniqueAVPacket        packet_        = nullptr;
    UniqueAVFrame         frame_         = nullptr;
    UniqueAVFrame         outFrame_      = nullptr;
    UniqueAVChannelLayout outChLayout_;

    AudioRingBuffer bufferL_;
    AudioRingBuffer bufferR_;

    [[nodiscard]] auto DecodeNextPacket() -> bool;
};

class AudioStreamWriter {
public:
    AudioStreamWriter(const std::string &path, int targetSr, const std::string &format);
    ~AudioStreamWriter();

    void Write(const float *L_data, const float *R_data, size_t length);
    void Close();

private:
    std::string path_;
    int         targetSr_;
    int64_t     ptsOffset_;

    UniqueAVFormatOutputContext formatCtx_ = nullptr;
    UniqueAVCodecContext        codecCtx_  = nullptr;
    AVStream                   *stream_    = nullptr;
    UniqueSwrContext            swrCtx_    = nullptr;
    UniqueAVFrame               encFrame_  = nullptr;
    UniqueAVPacket              packet_    = nullptr;

    AudioRingBuffer    bufferL_;
    AudioRingBuffer    bufferR_;
    std::vector<float> workBufL_;
    std::vector<float> workBufR_;

    void EncodeAndWrite(AVFrame *f);
    void ProcessInternalBuffer(bool flush = false);

    struct AudioChunk {
        std::vector<float> L;
        std::vector<float> R;
    };

    moodycamel::BlockingConcurrentQueue<AudioChunk> queue_;
    moodycamel::ConcurrentQueue<AudioChunk>         chunk_pool_;

    std::atomic<int>  queue_size_{ 0 };
    std::atomic<bool> stopFlag_{ false };
    std::thread       workerThread_;

    void WorkerLoop();
};