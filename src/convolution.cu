#include "convolution.hpp"
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdexcept>
#include <iostream>
__constant__ float d_kernel[25]; // max 5x5

void apply_gaussian(float *d_in, float *d_out, int width, int height, dim3 block, dim3 grid);

void apply_highpass(float *d_in, float *d_out, int width, int height, dim3 block, dim3 grid);

__global__ void convolve_gray(const float *in, float *out, int w, int h,
                              int ksize, float norm, OutputMode mode) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= w || y >= h)
    return;

  int r = ksize / 2;
  float sum = 0.0f;

  for (int ky = -r; ky <= r; ky++) {
    for (int kx = -r; kx <= r; kx++) {
      int ix = min(max(x + kx, 0), w - 1);
      int iy = min(max(y + ky, 0), h - 1);

      float k = d_kernel[(ky + r) * ksize + (kx + r)];
      sum += in[iy * w + ix] * k;
    }
  }
  float val;
  if (mode == ABS_CLAMP){
    val = fabsf(sum);
  } else{
    val = sum / norm;
  }
  val = fmin(fmax(val, 0.0f), 255.0f);
  out[y * w + x] = val;
}

void apply_filter(const Image &input, Image &output, FilterType filter) {
  int img_bytes = input.width * input.height * sizeof(float);
  float *d_in, *d_out;
  cudaMalloc(&d_in, img_bytes);
  cudaMalloc(&d_out, img_bytes);

  cudaMemcpy(d_in, input.data, img_bytes, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((input.width + 15) / 16, (input.height + 15) / 16);

  // Launch kernel
  if (filter == FilterType::GAUSSIAN) {
    apply_gaussian(d_in, d_out, input.width, input.height, block, grid);
  } else if (filter == FilterType::HIGHPASS) {
    apply_highpass(d_in, d_out, input.width, input.height, block, grid);
  }

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_in);
    cudaFree(d_out);
    std::cerr << "Kernel launch failed";
    throw std::runtime_error(cudaGetErrorString(err));
  }

  // Copy result back to host
  cudaMemcpy(output.data, d_out, img_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

void apply_gaussian(float *d_in, float *d_out, int width, int height, dim3 block, dim3 grid) {
  const float gaussian[25] = {1,  4, 7, 4,  1,  4,  16, 26, 16, 4, 7, 26, 41,
                              26, 7, 4, 16, 26, 16, 4,  1,  4,  7, 4, 1};
  int ksize = 5;
  float norm = 273.0f;

  cudaError_t err = cudaMemcpyToSymbol(d_kernel, gaussian, sizeof(gaussian));
  if (err != cudaSuccess) {
      std::cerr << "Failed to copy Highpass kernel to device: " << cudaGetErrorString(err) << std::endl;
      return;
  }
  OutputMode mode = NORMAL;

  convolve_gray<<<grid, block>>>(d_in, d_out, width, height, ksize, norm, mode);
}

void apply_highpass(float *d_in, float *d_out, int width, int height, dim3 block, dim3 grid) {
  const float highpass3x3[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
  int ksize = 3;
  float norm = 1.0f;

  cudaError_t err = cudaMemcpyToSymbol(d_kernel, highpass3x3, sizeof(highpass3x3));
  if (err != cudaSuccess) {
      std::cerr << "Failed to copy Highpass kernel to device: " << cudaGetErrorString(err) << std::endl;
      return;
  }
  OutputMode mode = ABS_CLAMP;

  convolve_gray<<<grid, block>>>(d_in, d_out, width, height, ksize, norm, mode);

  
}
