/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Blur_cuda.cuh"


__global__
void
cudaBlurKernel(const float *raw_data, const float *blur_v, float *out_data,
    int N, int blur_v_size) {

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < N) {
        for (int j = 0; j <= index && j < blur_v_size; j++){
            out_data[index] += raw_data[index - j] * blur_v[j];
        }
        index += blockDim.x * gridDim.x;
    }
}


void cudaCallBlurKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *raw_data,
        const float *blur_v,
        float *out_data,
        const unsigned int N,
        const unsigned int blur_v_size) {
        
    /* Call the kernel above this function. */
    cudaBlurKernel<<<blocks, threadsPerBlock>>>
        (raw_data, blur_v, out_data, N, blur_v_size);
}
