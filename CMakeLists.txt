cmake_minimum_required(VERSION 3.18)
project(photucuda LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 1. ENHANCE PORTABILITY: Static Linking
# This embeds the CUDA runtime into your binary so it runs on 
# distros that don't have the CUDA Toolkit installed.
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# 2. ENHANCE HARDWARE COMPATIBILITY: Multi-Arch Support
# This tells the compiler to generate code for multiple GPU generations.
# compute_75 = Turing (RTX 20)
# compute_80 = Ampere (A100)
# compute_86 = Ampere (RTX 30)
# compute_89 = Ada (RTX 40)
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