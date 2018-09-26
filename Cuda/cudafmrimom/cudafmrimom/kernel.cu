///////////////////////////////////////////////
//           File: kernel.cu                 //
///////////////////////////////////////////////
#include "declarations.h"

/* Creates an identity matrix with size row x col (needs 2D block) */
__global__ void initIdentityGPU(datatype *matrix, int row, int col) {
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if (x < row && y < col) {
		if (x == y) matrix[x*col + y] = 1.0;
		else matrix[x*col + y] = 0.0;
	}
}


/* Divides every element of the given matrix by a specific value */
__global__ void matrixDivNum(datatype *a, int size, datatype val) {
	int index = blockIdx.x *blockDim.x + threadIdx.x;
	if (index < size) a[index] = a[index] / val;
}


/* Subtracts two matrices a,b and stores the result in matrix c */
__global__ void subtractMatrices(datatype *c, datatype *a, datatype *b, int size) {
	int index = blockIdx.x *blockDim.x + threadIdx.x;
	if (index < size) c[index] = a[index] - b[index];
}


/* Adds two matrices a,b and stores the result in matrix c */
__global__ void addMatrices(datatype *c, datatype *a, datatype *b, int size) {
	int index = blockIdx.x *blockDim.x + threadIdx.x;
	if (index < size) c[index] = a[index] + b[index];
}


/* Performs soft thresholding to the input matrix arr
Y = sign(X)*(|X|-T)+ , soft thresholding is wavelet shrinkage ( (x)+ = 0 if x<0; (x)+ = x, if x >=0 ) */
__global__ void wthresh(datatype *arr, int size, datatype thres) {
	int index = blockIdx.x *blockDim.x + threadIdx.x;
	if (index < size) {
		datatype sign = 1.0;
		if (arr[index] < 0) {
			arr[index] = -arr[index];
			sign = -1.0;
		}
		arr[index] = arr[index] - thres;
		if (arr[index] < 0) arr[index] = 0;
		else arr[index] = sign *  arr[index];
	}
}


/* Divides each row of matrix b by the row matrix kv and stores result in matrix b */
__global__ void rdivide(datatype *b, int rowb, int colb, datatype *kv) {
	__shared__  datatype shm[K];
	int index = blockIdx.x *blockDim.x + threadIdx.x;
	if (threadIdx.x < K) shm[threadIdx.x] = kv[threadIdx.x];
	__syncthreads();
	if (index < rowb*colb) b[index] = b[index] / shm[index % colb];

}