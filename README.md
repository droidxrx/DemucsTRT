# DemucsTRT 🎵🚀

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-13.1-green.svg)
![TensorRT](https://img.shields.io/badge/TensorRT-Required-76b900.svg)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-007808.svg)

**DemucsTRT** is a high-performance C++ implementation for audio source separation (music demixing), accelerated by **NVIDIA TensorRT**. It provides low-latency streaming inference, making it ideal for real-time or rapid batch processing of audio stems (e.g., vocals, drums, bass, guitar, piano).

## ✨ Features

- **Blazing Fast Inference:** Leverages NVIDIA TensorRT for highly optimized GPU-accelerated execution.
- **Streaming Audio I/O:** Uses FFmpeg with lock-free concurrent queues to process audio streams efficiently without high memory overhead.
- **Dynamic & Static Shape Support:** Automatically detects and handles TensorRT models with dynamic batching.
- **Multi-Format Support:** Reads and writes various audio formats (WAV, MP3, AAC, FLAC, etc.) via FFmpeg.
- **Selective Stem Extraction:** Choose specific stems to extract to save I/O time (e.g., outputting only vocals).

## 📋 Prerequisites

Ensure you have the following dependencies installed on your system:
- **CMake** (v3.15+)
- **C++17** compatible compiler (MSVC, GCC, or Clang)
- **NVIDIA CUDA Toolkit** (Tested on v13.1)
- **NVIDIA TensorRT** (Tested on v11.0.0.114)
- **FFmpeg** (libavformat, libavcodec, libswresample, libavutil)

## 🛠️ Build Instructions

### 1. Configure the Environment
Create a `.env` file in the root directory of the project to specify your TensorRT installation path:

```env
TensorRT_ROOT=<path_to_your_tensorrt_installation>
```

### 2. Build the Project

Use CMake to configure and build the project:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

*Note: The build process automatically fetches `cxxopts` and `concurrentqueue` dependencies via CMake `FetchContent`.*

## 🧠 Model Preparation

Before running inference, you need to compile a TensorRT engine file (`.trt` or `.engine`) from a pre-trained ONNX model.

### 1. Download the ONNX Model

You can download the HTDemucs 6-stem ONNX model directly from Hugging Face.

```bash
# Create a models directory
mkdir models && cd models

# Download the ONNX model
wget https://huggingface.co/StemSplitio/htdemucs-6s-onnx/resolve/main/htdemucs_6s.onnx
```

### 2. Convert to TensorRT Engine

Use the `trtexec` command-line utility (included with your TensorRT installation) to compile the ONNX model into an optimized TensorRT engine.

```bash
# Note: Adjust the slashes depending on your OS (forward slashes for Linux, backslashes for Windows)
trtexec --onnx=models/htdemucs_6s.onnx --saveEngine=models/htdemucs_6s.trt --useCudaGraph
```

*(For Windows CMD/PowerShell, you can use: `trtexec --onnx=models\\htdemucs_6s.onnx --saveEngine=.\\models\\htdemucs_6s.trt --useCudaGraph`)*

## 🚀 Usage

Run the executable via the command line. The tool requires the compiled TensorRT engine and an input audio/video file.

```bash
./DemucsTRT --model path/to/model.trt [OPTIONS] path/to/song.mp3
```

### Command-Line Arguments

```
DemucsTRT [OPTION...] [file]

  -m, --model arg   [Required] Path to the TensorRT model file
  -o, --output arg  Output directory path. (default: stems)
  -f, --format arg  Output format audio (e.g., wav, mp3, flac). (default: wav)
  -s, --stems arg   Specific stems to extract (e.g., vocals, drums). (default: "all")
  -b, --batch arg   Batch size for inference (default: 1)
  -h, --help        Print usage instructions.
```

### Example

Extract only the vocals and drums from a song in FLAC format:

```bash
./DemucsTRT -m models/htdemucs_6s.trt -o output_dir -f flac -s "vocals,drums" -b 1 track.mp3
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing, ensure your code adheres to the included `.clang-format` and `.clang-tidy` guidelines.

## 🙏 Acknowledgements

* [Meta AI / Demucs](https://github.com/facebookresearch/demucs) — HTDemucs model, weights, and research
* [concurrentqueue](https://github.com/cameron314/concurrentqueue) — A fast multi-producer, multi-consumer lock-free concurrent queue for C++11
* [cxxopts](https://github.com/jarro2783/cxxopts) — Lightweight C++ command line option parser

## 📄 License

Code and tooling: MIT

HTDemucs model weights: [Meta AI license](https://github.com/facebookresearch/demucs/blob/main/LICENSE)