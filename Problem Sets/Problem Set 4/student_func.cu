//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

const int MAX_THREADS = 1024;

__global__
void hist_kernel(unsigned int * d_inputVals,
                 unsigned int * d_hist,
                 const size_t numElems,
                 const int bit)
{
  int globalPos = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalPos >= numElems) {
    return;
  }
  unsigned int one = 1;
  unsigned int val = ((d_inputVals[globalPos] & (one << bit)) == (one << bit)) ? 1 : 0;
  if (val == 1) {
    atomicAdd(&d_hist[1], 1);
  }
  else {
    atomicAdd(&d_hist[0], 1);
  }
}

__global__
void segment_scan_kernel(unsigned int* const d_inputVals,
                         unsigned int* d_scan,
                         const int numElems,
                         const int bit)
{
  int globalPos = blockIdx.x * blockDim.x + threadIdx.x;
  int localPos = threadIdx.x;

  __shared__ unsigned int sdata[MAX_THREADS];
  // Generate the array that is going to be scanned
  unsigned int one = 1;
  int val = 0;
  if (globalPos > 0 && globalPos < numElems) {
    val = ((d_inputVals[globalPos - 1] & (one << bit)) == (one << bit)) ? 1 : 0;
  }
  sdata[localPos] = val;
  __syncthreads();
  // Begin the scan on each segment
  for (int step = 1; step < MAX_THREADS; step *= 2) {
    if (localPos - step < 0) {
      d_scan[globalPos] = sdata[localPos];
      return;
    }
    unsigned int prev_val = sdata[localPos - step];
    __syncthreads();
    sdata[localPos] += prev_val;
    __syncthreads();
  }
  if (globalPos < numElems) {
    d_scan[globalPos] = sdata[localPos];
  }
}

__global__
void inclusive_scan_kernel(unsigned int* d_scan,
                           unsigned int* d_cumsum)
{
  // Copy the last element of each segment to the cumsum array
  int globalPos = threadIdx.x * MAX_THREADS + MAX_THREADS - 1;
  int localPos = threadIdx.x;
  d_cumsum[localPos] = d_scan[globalPos];
  __syncthreads();
  // Begin the scan of the cumsum array
  for (int step = 1; step < blockDim.x; step *= 2) {
    if (localPos - step < 0) {
      return;
    }
    unsigned int prev_val = d_cumsum[localPos - step];
    __syncthreads();
    d_cumsum[localPos] += prev_val;
    __syncthreads();
  }
}

__global__
void segment_sum_kernel(unsigned int* d_cumsum,
                        unsigned int* d_scan,
                        const int size,
                        const int numElems)
{
  int globalPos = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalPos >= numElems || blockIdx.x == 0) {
    return;
  }
  int cumsumVal = d_cumsum[blockIdx.x - 1];
  d_scan[globalPos] += cumsumVal;
}

__global__
void reorder_kernel(unsigned int* const d_inputVals,
                    unsigned int* const d_inputPos,
                    unsigned int* d_outputVals,
                    unsigned int* d_outputPos,
                    unsigned int* d_scan,
                    unsigned int* d_hist,
                    const int bit,
                    const size_t numElems)
{
  int globalPos = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalPos >= numElems) {
    return;
  }
  unsigned int one = 1;
  unsigned int outputPos;
  if ((d_inputVals[globalPos] & (one << bit)) == (one << bit)) {
    outputPos = d_scan[globalPos] + d_hist[0];
  }
  else {
    outputPos = globalPos - d_scan[globalPos];
  }
  d_outputVals[outputPos] = d_inputVals[globalPos];
  d_outputPos[outputPos] = d_inputPos[globalPos];
}

int debug = 1;
void debug_device_array(char* name, int l, unsigned int * d_arr, int numElems) {
    
   
    if(!debug)
        return;
    unsigned int h_arr[l];
    checkCudaErrors(cudaMemcpy(&h_arr, d_arr, l*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf(name);
    printf(" ");
    for(int i=0; i < l; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");
    unsigned int max = 0;
    unsigned int min = 1000000;
    unsigned int h_arr2[numElems];
    checkCudaErrors(cudaMemcpy(&h_arr2, d_arr, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    for(int i = 0; i < numElems; i++) {
        if(h_arr2[i] < min)
            min = h_arr2[i];
         if(h_arr2[i] > max)
            max = h_arr2[i];
    }
    printf("max %d min %d\n", max, min);
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
  // calculate histogram of 1/0 on each bit
  unsigned int * d_hist;
  checkCudaErrors(cudaMalloc(&d_hist, 2 * sizeof(unsigned int)));

  int blockNum = ceil((float)numElems / MAX_THREADS);
  const dim3 gridSize(blockNum, 1, 1);
  const dim3 blockSize(MAX_THREADS, 1, 1);
  const dim3 cumsumSize(blockNum - 1, 1, 1);
  const dim3 unitSize(1, 1, 1);

  unsigned int h_hist[2];

  // apply exclusive scan
  unsigned int * d_scan;
  unsigned int * d_cumsum;
  checkCudaErrors(cudaMalloc(&d_scan, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_cumsum, (blockNum - 1) * sizeof(unsigned int)));

  for (int i = 0; i < 32; i++) {
    // Reset temporary variables
    checkCudaErrors(cudaMemset(d_hist, 0, 2 * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_scan, 0, numElems * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_cumsum, 0, (blockNum - 1) * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_outputVals, 0, numElems * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_outputPos, 0, numElems * sizeof(unsigned int)));
    // Calculate the histogram of 1/0 on each bit
    hist_kernel<<<gridSize, blockSize>>>(d_inputVals, d_hist, numElems, i);
    // Use scan to calculate the 1 position
    segment_scan_kernel<<<gridSize, blockSize>>>(d_inputVals, d_scan, numElems, i);
    inclusive_scan_kernel<<<unitSize, cumsumSize>>>(d_scan, d_cumsum);
    segment_sum_kernel<<<gridSize, blockSize>>>(d_cumsum, d_scan, blockNum - 1, numElems);
    // start the sort for the ith bit
    reorder_kernel<<<gridSize, blockSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos,
                                            d_scan, d_hist, i, numElems);
    checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  }
  //debug_device_array("input", 50000, d_outputVals, numElems);
  checkCudaErrors(cudaFree(d_hist));
  checkCudaErrors(cudaFree(d_scan));
  checkCudaErrors(cudaFree(d_cumsum));

}
