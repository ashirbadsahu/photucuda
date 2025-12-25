cmake_minimum_required(VERSION 3.18)
project(photucuda LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

# Enable separable compilation (important for CUDA)
set_target_properties(photucuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(photucuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)
