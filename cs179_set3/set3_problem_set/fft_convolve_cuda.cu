/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>
#include <math.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < padded_length) {
        cufftComplex a = raw_data[index];
        cufftComplex b = impulse_v[index];
        cufftComplex c;
        c.x = ((a.x * b.x) - (a.y * b.y)) / padded_length;
        c.y = ((a.x * b.y) + (a.y * b.x)) / padded_length;
        
        out_data[index] = c;
        
        index += blockDim.x * gridDim.x;
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding and subsequent
    normalization (dividing by maximum).

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */
    
    // Create the shared memory.
    extern __shared__ float data[];
    
    // Figure out how many floats each thread will be handling.
    int numFloats = padded_length / (gridDim.x * blockDim.x) + 1;
    
    // Load the data from out_data into shared memory. Each thread only handles
    // numFloats sequential values.
    int index = blockIdx.x * blockDim.x * numFloats + threadIdx.x * numFloats;
    for (int j = 0; j < numFloats && index + j < padded_length; j++) {
        // We want the absolute value of out_data, not the complex value.
        float real = abs(out_data[index + j].x);
        if (j == 0) {
            data[threadIdx.x] = real;
        }
        else if(data[threadIdx.x] < real) {
            data[threadIdx.x] = real;
        }
    }
    
    __syncthreads();
    
    // Find the maximum value!
    // Set the initial stride length to be the number of threads in a block, so
    // that each thread will handle only numFloats values. This will be the case
    // in every iteration of the loop - every thread will only compare 
    // numFloats values.
    int strideLength = blockDim.x / 2;
    
    while (strideLength >= 1) {
        int ind = threadIdx.x;
        // If the thread index is less than the stride length, then continue.
        // This handles the "binary tree" reduction; each iteration, the number
        // of values that are compared is reduced by a factor of two, and as
        // each thread compares numFloats values, the number of threads that are
        // utilized is also reduced by a factor of two.
        if (ind < strideLength) {
            // Compare numFloats values, each one stride length apart. The max
            // value of these values will end up assigned to data[ind].
            if (data[ind] < data[ind + strideLength]) {
                data[ind] = data[ind + strideLength];
            }
        }
        
        // Cut the strideLength down by a factor of two.
        strideLength = strideLength / 2;
        __syncthreads();
    }

    // The maximum value over the section of out_data handled by this warp
    // should now be in data[0]. Use atomicMax to set the value of max_abs_val
    // appropriately.
    if (threadIdx.x == 0) {
        atomicMax(max_abs_val, data[0]);
    }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < padded_length) {
        out_data[index].x = out_data[index].x / *max_abs_val;
        out_data[index].y = out_data[index].y / *max_abs_val;
        
        index += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* TODO: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>
        (raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* TODO 2: Call the max-finding kernel. */
    cudaMaximumKernel<<< blocks, threadsPerBlock, threadsPerBlock * sizeof(float) >>>
        (out_data, max_abs_val, padded_length);

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>
        (out_data, max_abs_val, padded_length);
}
