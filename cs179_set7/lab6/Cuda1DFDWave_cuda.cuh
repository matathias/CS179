/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#ifndef CUDA_1D_FD_WAVE_CUDA_CUH
#define CUDA_1D_FD_WAVE_CUDA_CUH


/* TODO: This is a CUDA header file.
If you have any functions in your .cu file that need to be
accessed from the outside, declare them here */
void waveEquation(float *old_data, float *current_data, float *new_data,
                  int numberOfNodes, float constant, float left_boundary,
                  int blocks, int threadsPerBlock);

#endif // CUDA_1D_FD_WAVE_CUDA_CUH
