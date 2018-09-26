///////////////////////////////////////////////
//            File: cuLibs.cu                //
///////////////////////////////////////////////
#include <iostream>
#include <cusolverDn.h>
#include <curand.h>
#include "cublas_v2.h"
#include "declarations.h"
#include "errorTypes.h"
#include "mem.h"
#include INCLUDE_FILE(MATLAB_include,mat.h)
#include INCLUDE_FILE(MATLAB_include,matrix.h)
#include "primaryFunctions.h"


/* Calculates the frobenius norm of the given array */
datatype frobeniusNorm(cublasHandle_t handle, datatype *Array, int size) {
	datatype retval;
	cublasStatus_t stat;
	#if (MODE)
		if ((stat = cublasDnrm2(handle, size, Array, 1, &retval)) != CUBLAS_STATUS_SUCCESS) {
			printf("cublasDnmr2 failed: %s\n", cublasGetErrorString(stat));
			cleanup();
			exit(-1);
		}
	#else
		if ((stat = cublasSnrm2(handle, size, Array, 1, &retval)) != CUBLAS_STATUS_SUCCESS) {
			printf("cublasSnmr2 failed: %s\n", cublasGetErrorString(stat));
			cleanup();
			exit(-1);
		}
	#endif
	return retval;
}


/* Generate numbers using normal distribution
   http://docs.nvidia.com/cuda/curand/host-api-overview.html#axzz4SrFExYZh */
void randn(curandGenerator_t generator, datatype *Arr, int size) {
	curandStatus_t stat;
	#if (MODE)
		if ((stat = curandGenerateNormalDouble(generator, Arr, size, (datatype)0, (datatype)1)) != CURAND_STATUS_SUCCESS) {
			printf("curandGenerateNormalDouble failed: %s\n", curandGetErrorString(stat));
			cleanup();
			exit(-1);
		}
	#else
		if ((stat = curandGenerateNormal(generator, Arr, size, (datatype)0, (datatype)1)) != CURAND_STATUS_SUCCESS) {
			printf("curandGenerateNormal failed: %s\n", curandGetErrorString(stat));
			cleanup();
			exit(-1);
		}
	#endif
}


/* Returns the square root of the maximum eigenvalue of the given array using power method */
datatype findMaxSqrtEigenvalue(cusolverDnHandle_t cusolverH, datatype *dA, datatype *W, datatype *d_W, int *devInfo, const int m) {
	int lwork = 0;
	cudaError_t err;
	const int lda = m;
	datatype *d_work = 0;
	cusolverStatus_t stat;
	#if (MODE)
		if ((stat = cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, m, dA, lda, d_W, &lwork)) != CUSOLVER_STATUS_SUCCESS) {
			printf("cusolverDnDsyevd_bufferSize failed: %s\n", cusolverGetErrorString(stat));
			cleanup();
			exit(-1);
		}
	#else
		if ((stat = cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, m, dA, lda, d_W, &lwork)) != CUSOLVER_STATUS_SUCCESS) {
			printf("cusolverDnSsyevd_bufferSize failed: %s\n", cusolverGetErrorString(stat));
			cleanup();
			exit(-1);
		}
	#endif
	if ((err = cudaMalloc((void**)&d_work, lwork * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc d_work failed: %s\n", cudaGetErrorString(err));
		if (d_work && (err = cudaFree(d_work)) != cudaSuccess) printf("cudaFree d_work: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	#if (MODE)
		if ((stat = cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, m, dA, lda, d_W, d_work, lwork, devInfo)) != CUSOLVER_STATUS_SUCCESS) {
			printf("cusolverDnDsyevd failed: %s\n", cusolverGetErrorString(stat));
			if (d_work && (err = cudaFree(d_work)) != cudaSuccess) printf("cudaFree d_work: %s\n", cudaGetErrorString(err));
			cleanup();
			exit(-1);
		}
	#else
		if ((stat = cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, m, dA, lda, d_W, d_work, lwork, devInfo)) != CUSOLVER_STATUS_SUCCESS) {
			printf("cusolverDnSsyevd failed: %s\n", cusolverGetErrorString(stat));
			if (d_work && (err = cudaFree(d_work)) != cudaSuccess) printf("cudaFree d_work: %s\n", cudaGetErrorString(err));
			cleanup();
			exit(-1);
		}
	#endif	
	if ((err = cudaMemcpy(W, d_W, m * sizeof(datatype), cudaMemcpyDeviceToHost)) != cudaSuccess) {
		printf("cudaMemcpy W failed: %s\n", cudaGetErrorString(err));
		if (d_work && (err = cudaFree(d_work)) != cudaSuccess) printf("cudaFree d_work: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if (d_work && (err = cudaFree(d_work)) != cudaSuccess) printf("cudaFree d_work: %s\n", cudaGetErrorString(err));
	return sqrt(W[m - 1]);
}


/* Multiplies two matrices (category=0:C=A*B, category=1:C=A'*B, category=2:C=A*B') */
void gpu_blas_mmul(cublasHandle_t handle, const datatype *A, const datatype *B, datatype *C,
	int category, const int rowA, const int colA, const int rowB, const int colB) {

	int lda, ldb, ldc;
	const datatype alf = 1;
	const datatype bet = 0;
	const datatype *alpha = &alf;
	const datatype *beta = &bet;
	cublasStatus_t status;
	if (category == 0) { // C=A*B ( http://peterwittek.com/cublas-matrix-c-style.html )
		lda = colB; ldb = colA; ldc = colB;
		if (colA != rowB) {
			printf("colA != rowB. Cannot multiply the matrices\n");
			cleanup();
			exit(-1);
		}
		#if (MODE)
			status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, colB, rowA, colA, alpha, B, lda, A, ldb, beta, C, ldc);
		#else 
			status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, colB, rowA, colA, alpha, B, lda, A, ldb, beta, C, ldc);
		#endif
	}
	else if (category == 1) { // C = A'*B ( http://stackoverflow.com/questions/14595750/transpose-matrix-multiplication-in-cublas-howto )
		lda = colA; ldb = colB; ldc = colB;
		if (rowA != rowB) {
			printf("rowA != rowB. Cannot multiply the matrices\n");
			cleanup();
			exit(-1);
		}
		#if (MODE)
			status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, colB, colA, rowB, alpha, B, ldb, A, lda, beta, C, ldc);
		#else
			status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, colB, colA, rowB, alpha, B, ldb, A, lda, beta, C, ldc);
		#endif
	}
	else if (category == 2) { // C=A*B' 
		lda = colA; ldb = colB; ldc = rowB;
		if (colA != colB) {
			printf("colA != colB. Cannot multiply the matrices\n");
			cleanup();
			exit(-1);
		}
		#if (MODE)
			status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rowB, rowA, colB, alpha, B, ldb, A, lda, beta, C, ldc);
		#else
			status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, rowB, rowA, colB, alpha, B, ldb, A, lda, beta, C, ldc);
		#endif
	}
	else {
		printf("gpu_blas_mmul : not valid status %d \n", category);
		cleanup();
		exit(-1);
	}
	if (status != cudaSuccess) {
		printf("Cublas multiplication failed: %s\n", cublasGetErrorString(status));
		cleanup();
		exit(-1);
	}
}