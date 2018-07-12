/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

const int MAX_THREADS = 1024;
const int NUM_BINS = 1024;

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  int globalPos = blockIdx.x * blockDim.x + threadIdx.x;
  int localPos = threadIdx.x;

  __shared__ unsigned int sdata[NUM_BINS];
  if (localPos < NUM_BINS) {
    sdata[localPos] = 0;
  }
  __syncthreads();
  if (globalPos >= numVals) {
    return;
  }
  atomicAdd(&sdata[vals[globalPos]], 1);
  __syncthreads();
  if (localPos < NUM_BINS) {
    atomicAdd(&histo[localPos], sdata[localPos]);
  }

}

__global__
void naiveHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  int globalPos = blockDim.x * blockIdx.x + threadIdx.x;
  if (globalPos >= numVals) {
    return;
  }
  atomicAdd(&histo[vals[globalPos]], 1);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free
  int blockNum = ceil((float)numElems / MAX_THREADS);
  const dim3 gridSize(blockNum, 1, 1);
  const dim3 blockSize(MAX_THREADS, 1, 1);
  //naiveHisto<<<gridSize, blockSize>>>(d_vals, d_histo, numElems);
  yourHisto<<<gridSize, blockSize>>>(d_vals, d_histo, numElems); // optimized from 2.9msec to 0.36msec
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
