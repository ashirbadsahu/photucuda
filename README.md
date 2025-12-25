# Photucuda

GPU-accelerated image filters using CUDA and 2D convolution for high-performance image processing.

## Features

- **Gaussian Blur Filter**: Applies a Gaussian blur to reduce image noise and detail, commonly used for smoothing images.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 10.0 or later recommended)
- CMake (version 3.10 or later)
- C++ compiler (e.g., GCC, MSVC)
- Make or Ninja build system

## Installation

1. Clone the repository:
```bash
   git clone <repository-url>
   cd photucuda
```

2. Create a build directory and compile:
```bash
   mkdir build
   cd build
   cmake ..
   make
```

## Usage

Run the executable with an input image and a filter name:

```bash
./photucuda <input_image> <filter-name>
```

### Available Filters

- `gb`: Gaussian Blur

### Example

```bash
./photucuda input.png gb
```

This will apply the Gaussian blur filter to `input.png` and save the result as `output_gray.png`.

## License

This project is licensed under the MIT License.
