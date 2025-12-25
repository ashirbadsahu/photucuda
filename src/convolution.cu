#include "convolution.hpp"
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdexcept>

__constant__ float d_kernel[25]; // max 5x5

__global__ void convolve_gray(const float *in, float *out, int w, int h, int ksize,
                         float norm) {

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

  out[y * w + x] = sum / norm;
}

void apply_filter(const Image &input, Image &output, FilterType filter) {
  // Gaussian 5x5
  const float gaussian[25] = {1,  4, 7, 4,  1,  4,  16, 26, 16, 4, 7, 26, 41,
                              26, 7, 4, 16, 26, 16, 4,  1,  4,  7, 4, 1};

  int ksize = 5;
  float norm = 273.0f;

  cudaMemcpyToSymbol(d_kernel, gaussian, sizeof(gaussian));

  int img_bytes = input.width * input.height * sizeof(float);
  float *d_in, *d_out;
  cudaMalloc(&d_in, img_bytes);
  cudaMalloc(&d_out, img_bytes);

  cudaMemcpy(d_in, input.data, img_bytes, cudaMemcpyHostToDevice);

  // Launch kernel

  dim3 block(16, 16);
  dim3 grid((input.width + 15) / 16, (input.height + 15) / 16);

  convolve_gray<<<grid, block>>>(d_in, d_out, input.width, input.height, 5, 273.0f);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_in);
    cudaFree(d_out);
    throw std::runtime_error(cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(output.data, d_out, img_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}
