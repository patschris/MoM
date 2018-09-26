///////////////////////////////////////////////
//           File: reduction.cu              //
///////////////////////////////////////////////
#include <math.h>
#include "declarations.h"
#include "reduction.h"
#include "Globals.h"


// Macro for finding the minimum of 2 numbers
#define MINIMUM(a,b) ( a < b ? a : b )

// Temporary reduction storage buffer
__device__ datatype experimental_array[K * 1024];


/* Action performed on data before reduction of sum of elements! 'SUM OF SQUARE VALUES' */
__inline__ __device__ datatype sqval(datatype val) {
	return val*val;
}


/* Reduction within a single warp */
__inline__ __device__ datatype warpReduceSum(datatype val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}


/* Reduction within a single block */
__inline__ __device__ datatype blockReduceSum(datatype val) {
	// Max. block size = 1024 (32 x 32) = 32 warps of 32 threads 
	static __shared__ datatype shared[32];	// Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	val = warpReduceSum(val);	// Each warp performs partial reduction
	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory
	__syncthreads(); // Wait for all partial reductions
	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;					
	if (wid == 0) val = warpReduceSum(val);	//Final reduce within first warp 
	return val;
}


/* Reduction across multiple blocks -> STAGE 1: Blocks to buffer */
__global__ void deviceReduceKernel(datatype *in, int N, int cols) {
	datatype sum = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < N){
		// Retrieve value filtered through function
		sum = sqval(in[i*cols + blockIdx.y]);
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		// Write to first stage buffer
		experimental_array[blockIdx.y * 1024 + blockIdx.x] = sum;
	}
}


/* Reduction across multiple blocks -> STAGE 2: Partial sums */
__global__ void deviceReduceKernel2(datatype* out, int N) {
	datatype sum = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N){
		// Retrieve partial sum
		sum = *(experimental_array + blockIdx.y * 1024 + i);
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		// Write to output...
		datatype s = sqrt(sum);
		// ...after Soft Thresholding
		out[blockIdx.y] = (s < CCL ? 1.0 : s);
	}
}


/* Normalization of columns of A by 2 stage square sum reductions */
void normalizeReduction(datatype *A, datatype* B) {
	dim3 block(512);
	dim3 grid(MINIMUM((Globals::rowsX + block.x - 1) / block.x, 1024), K);
	// 1st stage
	deviceReduceKernel << <grid, block >> > (A, Globals::rowsX, K);
	dim3 grid2(1, K);
	// 2nd stage
	deviceReduceKernel2 << <grid2, 1024 >> > (B, grid.x);
}