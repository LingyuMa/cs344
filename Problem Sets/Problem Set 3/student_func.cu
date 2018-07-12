/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

const int MAX_THREADS = 1024;

const float LOWER_BOUND = 0;
const float UPPER_BOUND = 1000;

__global__
void reduce_min_max(const float* d_in,
                    float* d_out,
                    int arrSize,
                    bool isMin)
{
  extern __shared__ float sdata[];
  int globalPos = blockIdx.x * MAX_THREADS + threadIdx.x;
  int localPos = threadIdx.x;

  if (globalPos < arrSize) {
    sdata[localPos] = d_in[globalPos];
  }
  else {
    sdata[localPos] = isMin ? UPPER_BOUND : LOWER_BOUND;
  }
  __syncthreads();

  for (int len = blockDim.x / 2; len > 0; len = len >> 1) {
    if (localPos < len) {
      sdata[localPos] = isMin ? min(sdata[localPos + len], sdata[localPos]) : max(sdata[localPos + len], sdata[localPos]);
    }
    __syncthreads();
  }

  if (localPos == 0) {
    d_out[blockIdx.x] = sdata[0];
  }
}

__global__
void naive_hist_kernel(const float* const d_logLuminance,
                       unsigned int* d_hist,
                       const size_t numRows,
                       const size_t numCols,
                       const size_t numBins,
                       const float min_logLum,
                       const float max_logLum)
{
  int globalPos = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalPos >= numRows * numCols) {
    return;
  }
  float lum_range = max_logLum - min_logLum;
  int bin = ((d_logLuminance[globalPos] - min_logLum) / lum_range) * numBins;
  if (bin == numBins) {
    bin -= 1;
  }
  atomicAdd(&d_hist[bin], 1);
}

__global__
void hist_kernel(const float* const d_logLuminance,
                 unsigned int* localHist,
                 const size_t numRows,
                 const size_t numCols,
                 const size_t numBins,
                 const float min_logLum,
                 const float max_logLum)
{
  int globalPos = blockIdx.x * blockDim.x + threadIdx.x;
  int localPos = threadIdx.x;
  if (globalPos >= numRows * numCols) {
    return;
  }
  float lum_range = max_logLum - min_logLum;
  int bin = ((d_logLuminance[globalPos] - min_logLum) / lum_range) * numBins;
  if (bin == numBins) {
    bin -= 1;
  }
  atomicAdd(&localHist[blockIdx.x * numBins + bin], 1);
}

__global__
void hist_reduce_kernel(const unsigned int* const localHist,
                        unsigned int* d_hist,
                        const int histCount,
                        const int numBins)
{
  int localPos = threadIdx.x;
  unsigned int res = 0;
  for (int i = 0; i < histCount; i++) {
    res += localHist[localPos + i * numBins];
  }
  d_hist[localPos] = res;
}

__global__
void scan_kernel(unsigned int* d_hist,
                 const int numBins)
{
  int globalPos = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalPos >= numBins) {
    return;
  }
  for (int i = 1; i <= numBins / 2; i *= 2) {
    if (globalPos - i < 0) {
      return;
    }
    int tmp = d_hist[globalPos - i];
    __syncthreads();
    d_hist[globalPos] += tmp;
    __syncthreads();
  }
}

void reduce(const float* const d_logLuminance, 
            float &min_logLum, 
            float &max_logLum, 
            const size_t numRows, 
            const size_t numCols)
{
  int blockNum = numRows * numCols / MAX_THREADS + 1;
  const dim3 GlobalReduceGridSize(blockNum, 1, 1);
  const dim3 GlobalReduceBlockSize(MAX_THREADS, 1, 1);
  const dim3 LocalReduceGridSize(1, 1, 1);
  const dim3 LocalReduceBlockSize(blockNum, 1, 1);

  float* d_intermediate;
  float* d_min;
  float* d_max;

  checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float) * blockNum));
  checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
  checkCudaErrors(cudaMemset(d_intermediate, LOWER_BOUND, sizeof(float) * blockNum));

  reduce_min_max<<<GlobalReduceGridSize, GlobalReduceBlockSize, sizeof(float) * MAX_THREADS>>>(d_logLuminance,
                                                                                               d_intermediate,
                                                                                               numRows * numCols,
                                                                                               false);
  reduce_min_max<<<LocalReduceGridSize, LocalReduceBlockSize, sizeof(float) * blockNum>>>(d_intermediate,
                                                                                          d_max,
                                                                                          blockNum,
                                                                                          false);
  float h_max;
  checkCudaErrors(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
  max_logLum = h_max;

  checkCudaErrors(cudaMemset(d_intermediate, UPPER_BOUND, sizeof(float) * blockNum));
  reduce_min_max<<<GlobalReduceGridSize, GlobalReduceBlockSize, sizeof(float) * MAX_THREADS>>>(d_logLuminance,
                                                                                              d_intermediate,
                                                                                              numRows * numCols,
                                                                                              true);
  reduce_min_max<<<LocalReduceGridSize, LocalReduceBlockSize, sizeof(float) * blockNum>>>(d_intermediate,
                                                                                          d_min,
                                                                                          blockNum,
                                                                                          true);
  float h_min;
  checkCudaErrors(cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
  min_logLum = h_min;
  checkCudaErrors(cudaFree(d_intermediate));
  checkCudaErrors(cudaFree(d_min));
  checkCudaErrors(cudaFree(d_max));
}

void hist(const float* const d_logLuminance,
          unsigned int* d_hist,
          const size_t numRows,
          const size_t numCols,
          const size_t numBins,
          const float min_logLum,
          const float max_logLum)
{
  int blockNum = numRows * numCols / MAX_THREADS + 1;
  const dim3 gridDim(blockNum, 1, 1);
  const dim3 blockDim(MAX_THREADS, 1, 1);
  // naive implementation cost 0.53 msecs
  unsigned int* d_localHist;
  checkCudaErrors(cudaMalloc(&d_localHist, sizeof(unsigned int) * numBins * blockNum));
  checkCudaErrors(cudaMemset(d_localHist, 0, sizeof(unsigned int) * numBins * blockNum));

  // naive_hist_kernel<<<gridDim, blockDim>>>(d_logLuminance,
  //                                    d_hist,
  //                                    numRows,
  //                                    numCols,
  //                                    numBins,
  //                                    min_logLum,
  //                                    max_logLum);

  hist_kernel<<<gridDim, blockDim>>>(d_logLuminance,
                                     d_localHist,
                                     numRows,
                                     numCols,
                                     numBins,
                                     min_logLum,
                                     max_logLum);
  const dim3 reduceGridDim(1, 1, 1);
  const dim3 reduceBlockDim(numBins, 1, 1);
  hist_reduce_kernel<<<reduceGridDim, reduceBlockDim>>>(d_localHist, 
                                                        d_hist, 
                                                        blockNum, 
                                                        numBins);
  checkCudaErrors(cudaFree(d_localHist));
}

void scan(unsigned int* d_hist, unsigned int* const d_cdf, const size_t numBins)
{
  int blockNum = std::ceil((float)numBins / MAX_THREADS);
  const dim3 gridDim(blockNum, 1, 1);
  const dim3 blockDim(MAX_THREADS, 1, 1);
  scan_kernel<<<gridDim, blockDim>>>(d_hist, numBins);
  checkCudaErrors(cudaMemcpy(d_cdf, d_hist, numBins * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(d_hist));
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  // Find minimum and maximum value of the image (Assume total pixel < 1024 x 1024)
  reduce(d_logLuminance, min_logLum, max_logLum, numRows, numCols);
  std::cout << "minimum luminance: " << min_logLum << std::endl;
  std::cout << "maximum luminance: " << max_logLum << std::endl;

  // Generate the histogram
  unsigned int* d_hist;
  checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) * numBins));

  hist(d_logLuminance, d_hist, numRows, numCols, numBins, min_logLum, max_logLum);

  // unsigned int h_out[100];
  // cudaMemcpy(&h_out, d_hist, sizeof(unsigned int) * 100, cudaMemcpyDeviceToHost);
  // for(int i = 0; i < 100; i++)
  // std::cout << "hist out: " << h_out[i] << std::endl;

  scan(d_hist, d_cdf, numBins);

}
