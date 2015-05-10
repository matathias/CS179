
/* 
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>

#define PI 3.14159265358979


/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}

/* Basic ramp filter. Scale all frequencies linearly. */
__global__ void cudaFrequencyKernal(cufftComplex *out_data, int length) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < length) {
        float scaleFactor;
        // We need to account for the fact that the highest amplitude is at
        // length / 2
        if (index < (length / 2)) {
            scaleFactor = ((float) index) / (length / 2); 
        }
        else {
            scaleFactor = ((float) (length - index)) / (length / 2);
        }
        cufftComplex temp = out_data[index];
        temp.x = temp.x * scaleFactor;
        temp.y = temp.y * scaleFactor;
        
        out_data[index] = temp;
        
        index += blockDim.x * gridDim.x;
    }
}

/* Convert an array of complex values to an array of real values. */
__global__ void cudaComplexToRealKernal(cufftComplex *in_data,
                                         float *out_data,
                                         int length) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < length) {
        cufftComplex in = in_data[index];
        
        out_data[index] = in.x;
        
        index += blockDim.x * gridDim.x;
    }
}

/* Backproject the sinogram to an image. */
__global__ void cudaBackprojectionKernal(float *in_data, float *out_data,
                                          int nAngles, int sin_width,
                                          int image_dim) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < (image_dim * image_dim)) {
        // Get the pixel (x,y) coordinate from the index value
        int y_image = index / image_dim;
        int x_image = index % image_dim;
        // Get the geometric (x,y) coordinate from the pixel coordinate
        int x_geo = x_image - (image_dim / 2);
        int y_geo = (image_dim / 2) - y_image;
        
        // For all theta in the sinogram...
        for (int i = 0; i < nAngles; i++) {
            float d;
            // Handle the edges cases of theta = 0 and theta = PI/2
            if(i == 0) {
                d = (float) x_geo;
            }
            else if (i == nAngles / 2) {
                d = (float) y_geo;
            }
            else {
                float theta = PI * (((float) i) / ((float) nAngles));
                float m = -1 * cos(theta) / sin(theta);
                float x_i = ((float) (y_geo - m * x_geo)) / ((-1 / m) - m);
                float y_i = (-1 / m) * x_i;
                d = sqrt((x_i * x_i) + (y_i * y_i));
                // Center the index
                if (((-1 / m) > 0 && x_i < 0) || ((-1 / m) < 0 && x_i > 0)) {
                    d *= -1;
                }
            }
            // d is the distance from the center line, so we need to offset d by
            // this much
            d += sin_width / 2.0;
            d = truncf(d);
            // Now that we have d, add the right value to the image array
            out_data[y_image * image_dim + x_image] += in_data[i * sin_width + (int)d];
        }
        
        index += blockDim.x * gridDim.x;
    }
}




int main(int argc, char** argv){

    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Sinogram filename > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output filename >\n");
        exit(EXIT_FAILURE);
    }


    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float; 
    float* output_dev;  // Image storage


    cufftComplex *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);




    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftComplex *)malloc(  sinogram_width*nAngles*sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    /* TODO: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    int sinogram_size = nAngles * sinogram_width;
    cudaMalloc((void **) &dev_sinogram_cmplx, sizeof(cufftComplex) * sinogram_size);
    cudaMalloc((void **) &dev_sinogram_float, sizeof(float) * sinogram_size);
               
    cudaMemcpy(dev_sinogram_cmplx, sinogram_host, 
               sizeof(cufftComplex) * sinogram_size, 
               cudaMemcpyHostToDevice);


    /* TODO 1: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */
    cufftHandle plan;
    int batch = 1;
    cufftPlan1d(&plan, sinogram_size, CUFFT_C2C, batch);
    
    // Run the forward DFT
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD);
    
    // Apply basic ramp filter
    cudaFrequencyKernal<<<nBlocks, threadsPerBlock>>>
                        (dev_sinogram_cmplx, sinogram_size);
    
    // Run the inverse DFT
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_INVERSE);
    
    // Extract the real components to floats
    cudaComplexToRealKernal<<<nBlocks, threadsPerBlock>>>
                        (dev_sinogram_cmplx, dev_sinogram_float, sinogram_size);
                        
    // Free the original sinogram
    cudaFree(dev_sinogram_cmplx);


    /* TODO 2: Implement backprojection.
        - Allocate memory for the output image.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */
    cudaMalloc((void **) &output_dev, sizeof(float) * width * height);
    cudaMemset(output_dev, 0, sizeof(float) * width * height);
    
    // Run the Backprojection kernal
    cudaBackprojectionKernal<<<nBlocks, threadsPerBlock>>>(dev_sinogram_float,
                                                           output_dev, 
                                                           nAngles,
                                                           sinogram_width, 
                                                           width);
                                                           
    // Copy the reconstructed image back to host
    cudaMemcpy(output_host, output_dev, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    
    // Free the remaining GPU memory
    cudaFree(dev_sinogram_float);
    cudaFree(output_dev);

    
    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);

    return 0;
}



