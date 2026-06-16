# ~~~
# FindTensorRT Merge and Optimization Module
#
# This module defines the following variables:
# - TensorRT_FOUND            : A Boolean that determines whether TensorRT was found.
# - TensorRT_VERSION          : The detected TensorRT version (Major.Minor.Patch).
# - TensorRT_INCLUDE_DIRS     : The TensorRT header directory.
# - TensorRT_LIBRARIES        : A list of full paths to TensorRT libraries.
#
# This module automatically builds the following Modern CMake Targets:
# - trt::nvinfer
# - trt::nvinfer_plugin
# - trt::nvonnxparser
# - trt::nvparsers            (Only if TensorRT version < 9)
# - trt::tensorrt             (Meta-target that links to all the above components)
# ~~~

include(FindPackageHandleStandardArgs)

# 1. Specify a search directory (HINTS)
if(NOT DEFINED TensorRT_ROOT)
    if(DEFINED ENV{TensorRT_ROOT})
        set(TensorRT_ROOT $ENV{TensorRT_ROOT})
    elseif(DEFINED ENV{TENSORRT_ROOT})
        set(TensorRT_ROOT $ENV{TENSORRT_ROOT})
    else()
        set(TensorRT_ROOT "/usr" CACHE PATH "Folder instalasi NVIDIA TensorRT")
    endif()
endif()

set(_TENSORRT_HINTS
    ${TensorRT_ROOT}
    ${TENSORRT_ROOT}
    ${CUDA_TOOLKIT_ROOT_DIR}
)

# 2. Search the Include Directory
find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_TENSORRT_HINTS}
    PATH_SUFFIXES include
)

# 3. Extract the TensorRT Version
if(TensorRT_INCLUDE_DIR)
    # Version headers are separated starting with a specific TensorRT version
    if(EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
        file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _TRT_VERSION_CONTENTS)
    elseif(EXISTS "${TensorRT_INCLUDE_DIR}/NvInfer.h")
        file(READ "${TensorRT_INCLUDE_DIR}/NvInfer.h" _TRT_VERSION_CONTENTS)
    endif()

    if(_TRT_VERSION_CONTENTS)
        # Match the standard NV_TENSORRT_* format
        string(REGEX MATCH "define[ \t]+NV_TENSORRT_MAJOR[ \t]+([0-9]+)" _ "${_TRT_VERSION_CONTENTS}")
        set(TensorRT_VERSION_MAJOR "${CMAKE_MATCH_1}")

        string(REGEX MATCH "define[ \t]+NV_TENSORRT_MINOR[ \t]+([0-9]+)" _ "${_TRT_VERSION_CONTENTS}")
        set(TensorRT_VERSION_MINOR "${CMAKE_MATCH_1}")

        string(REGEX MATCH "define[ \t]+NV_TENSORRT_PATCH[ \t]+([0-9]+)" _ "${_TRT_VERSION_CONTENTS}")
        set(TensorRT_VERSION_PATCH "${CMAKE_MATCH_1}")

        # If not found (TRT 10.11+), use TRT_*_ENTERPRISE
        if(NOT TensorRT_VERSION_MAJOR)
            string(REGEX MATCH "define[ \t]+TRT_MAJOR_ENTERPRISE[ \t]+([0-9]+)" _ "${_TRT_VERSION_CONTENTS}")
            set(TensorRT_VERSION_MAJOR "${CMAKE_MATCH_1}")

            string(REGEX MATCH "define[ \t]+TRT_MINOR_ENTERPRISE[ \t]+([0-9]+)" _ "${_TRT_VERSION_CONTENTS}")
            set(TensorRT_VERSION_MINOR "${CMAKE_MATCH_1}")

            string(REGEX MATCH "define[ \t]+TRT_PATCH_ENTERPRISE[ \t]+([0-9]+)" _ "${_TRT_VERSION_CONTENTS}")
            set(TensorRT_VERSION_PATCH "${CMAKE_MATCH_1}")
        endif()

        if(TensorRT_VERSION_MAJOR)
            set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}"
                CACHE STRING "Terdeteksi versi TensorRT"
            )
        endif()
    endif()
    unset(_TRT_VERSION_CONTENTS)
endif()

# 4. Filter Components by Version
if(NOT TensorRT_FIND_COMPONENTS)
    set(TensorRT_FIND_COMPONENTS nvinfer nvinfer_plugin nvonnxparser)
    
    # nvparsers is deprecated and removed in TensorRT 9 and above
    if(TensorRT_VERSION_MAJOR VERSION_LESS 9)
        list(APPEND TensorRT_FIND_COMPONENTS nvparsers)
    endif()
endif()

set(TensorRT_LIBRARIES "")

# 5. Search Functions Libraries and Building Targets
function(_find_trt_component component)
    find_library(
        TensorRT_${component}_LIBRARY
        # Add support for library name variations on Windows (e.g., nvinfer_11, nvinfer64_11)
        NAMES 
            ${component} 
            ${component}_${TensorRT_VERSION_MAJOR} 
            ${component}64_${TensorRT_VERSION_MAJOR}
        HINTS 
            ${_TENSORRT_HINTS}
            ${CUDAToolkit_LIBRARY_DIR} # Use hints from the modern CUDAToolkit
        PATH_SUFFIXES 
            lib lib64 lib/x64 bin
    )

    if(TensorRT_${component}_LIBRARY)
        set(TensorRT_${component}_FOUND TRUE CACHE INTERNAL "Found ${component}")
        
        # Include in the LIBRARIES list using the Parent Scope reference variable
        set(_TMP_LIBS ${TensorRT_LIBRARIES})
        list(APPEND _TMP_LIBS ${TensorRT_${component}_LIBRARY})
        set(TensorRT_LIBRARIES "${_TMP_LIBS}" PARENT_SCOPE)

        # Create a Modern CMake Target
        if(NOT TARGET trt::${component})
            add_library(trt::${component} UNKNOWN IMPORTED)
            set_target_properties(trt::${component} PROPERTIES
                IMPORTED_LOCATION "${TensorRT_${component}_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
            )
        endif()
    else()
        set(TensorRT_${component}_FOUND FALSE CACHE INTERNAL "Not Found ${component}")
    endif()
endfunction()

# Iterate through each component and perform a search
foreach(component IN LISTS TensorRT_FIND_COMPONENTS)
    _find_trt_component(${component})
endforeach()

# 6. Validate CMake Standard Args
find_package_handle_standard_args(TensorRT
    FOUND_VAR TensorRT_FOUND
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_LIBRARIES
    VERSION_VAR TensorRT_VERSION
    HANDLE_COMPONENTS
)

# 7. Export Final Variables and Meta-Targets
if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
    
    # Create meta-targets to simplify linking (optional but recommended)
    if(NOT TARGET trt::tensorrt)
        add_library(trt::tensorrt INTERFACE IMPORTED)
        foreach(component IN LISTS TensorRT_FIND_COMPONENTS)
            if(TARGET trt::${component})
                target_link_libraries(trt::tensorrt INTERFACE trt::${component})
            endif()
        endforeach()
    endif()
endif()

# Hide internal cache variables from the CMake GUI
mark_as_advanced(
    TensorRT_INCLUDE_DIR
    TensorRT_LIBRARIES
    TensorRT_VERSION
)