cmake_minimum_required(VERSION 3.18)
project(photucuda LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# ENHANCE HARDWARE COMPATIBILITY: Multi-Arch Support

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

add_executable(photucuda
    src/main.cpp
    src/image_io.cpp
    src/convolution.cu
)

# Include paths
target_include_directories(photucuda PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/stb
)

# Enable separable compilation
set_target_properties(photucuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optimization and Warning Flags
target_compile_options(photucuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)