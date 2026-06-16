cmake_minimum_required(VERSION 3.15)
project(DemucsTRT)

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(LoadEnv)

load_env_file("${CMAKE_CURRENT_SOURCE_DIR}/.env" REQUIRED)

find_package(CUDAToolkit 13.1 REQUIRED)
find_package(TensorRT REQUIRED)

message(STATUS "Found TensorRT version ${TensorRT_VERSION}")
message(STATUS "Found TensorRT include dir at ${TensorRT_INCLUDE_DIR}")
message(STATUS "Found TensorRT libs at ${TensorRT_LIBRARIES}")

include(FetchContent)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        b6135315a54deb2b556219906eda8346f69bf703
    GIT_SHALLOW    ON
)

FetchContent_Declare(
    concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG        v1.0.5
    GIT_SHALLOW    ON
)

FetchContent_MakeAvailable(cxxopts concurrentqueue)

add_executable(DemucsTRT src/core.cpp src/demucs.cpp src/AudioStream.cpp)
target_compile_definitions(DemucsTRT PRIVATE NOMINMAX)
target_link_libraries(DemucsTRT PRIVATE CUDA::cudart trt::nvinfer cxxopts concurrentqueue)

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_BUILD_TYPE)

if(uppercase_BUILD_TYPE STREQUAL "DEBUG")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(DemucsTRT PRIVATE /fsanitize=address /Zi /INCREMENTAL:NO)
        target_link_options(DemucsTRT PRIVATE /fsanitize=address)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(DemucsTRT PRIVATE -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
        target_link_options(DemucsTRT PRIVATE -fsanitize=address -fsanitize=undefined)
    endif()
endif()

if(WIN32)
    find_program(FFMPEG_EXECUTABLE ffmpeg)
    if(NOT FFMPEG_EXECUTABLE)
        message(FATAL_ERROR "FFMPEG not found. Please install FFMPEG and ensure it's in your PATH.")
    endif()
    message(STATUS "Found FFMPEG executable at ${FFMPEG_EXECUTABLE}")

    cmake_path(GET FFMPEG_EXECUTABLE PARENT_PATH FFMPEG_BIN_DIR)
    set(FFMPEG_INCLUDE_DIR "${FFMPEG_BIN_DIR}/../include")
    set(FFMPEG_LIB_DIR "${FFMPEG_BIN_DIR}/../lib")

    target_include_directories(DemucsTRT PRIVATE ${FFMPEG_INCLUDE_DIR})
    target_link_directories(DemucsTRT PRIVATE ${FFMPEG_BIN_DIR} ${FFMPEG_LIB_DIR})
    target_link_libraries(DemucsTRT PRIVATE avformat avcodec swresample avutil)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libswresample libavutil)
    target_link_libraries(DemucsTRT PRIVATE PkgConfig::FFMPEG)
endif()