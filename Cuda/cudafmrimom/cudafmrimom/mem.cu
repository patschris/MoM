///////////////////////////////////////////////
//              File: mem.cu                 //
///////////////////////////////////////////////
#include <iostream>
#include <ctime>
#include <curand.h>
#include <cusolverDn.h>
#include "cublas_v2.h"
#include "declarations.h"
#include "errorTypes.h"
#include "Globals.h"

///////////////////////////////////////////////////////////////////////////////
/* Global variables */
extern int *devInfo;
extern cublasHandle_t handle;
extern curandGenerator_t generator;
extern cusolverDnHandle_t cusolverH;
extern datatype *I, *KK, *KK2, *KVOXELS, *KVOXELS2, *TIMECOMPK, *TIMECOMPK2, *Kv, *TIMECOMPVOXELS, *X, *D, *S, *W, *d_W;
///////////////////////////////////////////////////////////////////////////////


/* Free all the previously allocated memory */
void cleanup() {
	cudaError_t err;
	curandStatus_t curstat;
	cublasStatus_t cubstat;
	cusolverStatus_t cusstat;
	if (W) free(W);
	if (generator && (curstat = curandDestroyGenerator(generator)) != CURAND_STATUS_SUCCESS)
		printf("curandDestroyGenerator failed: %s\n", curandGetErrorString(curstat));
	if (handle && (cubstat = cublasDestroy(handle)) != CUBLAS_STATUS_SUCCESS)
		printf("cublasDestroy failed: %s\n", cublasGetErrorString(cubstat));
	if (X && (err = cudaFree(X)) != cudaSuccess) printf("cudaFree X failed: %s\n", cudaGetErrorString(err));
	if (D && (err = cudaFree(D)) != cudaSuccess) printf("cudaFree D failed: %s\n", cudaGetErrorString(err));
	if (S && (err = cudaFree(S)) != cudaSuccess) printf("cudaFree S failed: %s\n", cudaGetErrorString(err));
	if (I && (err = cudaFree(I)) != cudaSuccess) printf("cudaFree cuI failed: %s\n", cudaGetErrorString(err));
	if (KK && (err = cudaFree(KK)) != cudaSuccess) printf("cudaFree KK failed: %s\n", cudaGetErrorString(err));
	if (KK2 && (err = cudaFree(KK2)) != cudaSuccess) printf("cudaFree KK2 failed: %s\n", cudaGetErrorString(err));
	if (KVOXELS && (err = cudaFree(KVOXELS)) != cudaSuccess) printf("cudaFree KVOXELS failed: %s\n", cudaGetErrorString(err));
	if (KVOXELS2 && (err = cudaFree(KVOXELS2)) != cudaSuccess) printf("cudaFree KVOXELS2 failed: %s\n", cudaGetErrorString(err));
	if (TIMECOMPK && (err = cudaFree(TIMECOMPK)) != cudaSuccess) printf("cudaFree TIMECOMPK failed: %s\n", cudaGetErrorString(err));
	if (TIMECOMPK2 && (err = cudaFree(TIMECOMPK2)) != cudaSuccess) printf("cudaFree TIMECOMPK2 failed: %s\n", cudaGetErrorString(err));
	if (Kv && (err = cudaFree(Kv)) != cudaSuccess) printf("cudaFree Kv failed: %s\n", cudaGetErrorString(err));
	if (TIMECOMPVOXELS && (err = cudaFree(TIMECOMPVOXELS)) != cudaSuccess) printf("cudaFree TIMECOMPVOXELS failed: %s\n", cudaGetErrorString(err));
	if (devInfo && (err = cudaFree(devInfo)) != cudaSuccess) printf("cudaFree devInfo failed: %s\n", cudaGetErrorString(err));
	if (cusolverH && (cusstat = cusolverDnDestroy(cusolverH)) != CUSOLVER_STATUS_SUCCESS)
		printf("cusolverDnDestroy failed: %s\n", cusolverGetErrorString(cusstat));
	if (d_W && (err = cudaFree(d_W)) != cudaSuccess) printf("cudaFree d_W failed: %s\n", cudaGetErrorString(err));
	if (cudaDeviceReset() != cudaSuccess) printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
}


/* Allocate all necessary memory for global variables */
void allocate() {
	cudaError_t err;
	cublasStatus_t cubStat;
	curandStatus_t curStat;
	cusolverStatus_t cusStat;
	if ((W = (datatype *)malloc(K * sizeof(datatype))) == NULL) {
		perror("malloc failed W");
		cleanup();
		exit(-1);
	}
	if ((cubStat = cublasCreate(&handle)) != CUBLAS_STATUS_SUCCESS) {
		printf("cublasCreate failed: %s\n", cublasGetErrorString(cubStat));
		cleanup();
		exit(-1);
	}
	if ((cusStat = cusolverDnCreate(&cusolverH)) != CUSOLVER_STATUS_SUCCESS) {
		printf("cusolverDnCreate failed: %s\n", cusolverGetErrorString(cusStat));
		cleanup();
		exit(-1);
	}
	if ((curStat = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT)) != CURAND_STATUS_SUCCESS) {
		printf("curandCreateGenerator failed: %s\n", curandGetErrorString(curStat));
		cleanup();
		exit(-1);
	}
	if ((curStat = curandSetPseudoRandomGeneratorSeed(generator, (unsigned long long) clock())) != CURAND_STATUS_SUCCESS) {
		printf("curandSetPseudoRandomGeneratorSeed failed: %s\n", curandGetErrorString(curStat));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&X, Globals::rowsX * Globals::colsX * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed cuX: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&D, Globals::rowsX * K * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed D: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&S, K * Globals::colsX * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed S: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&I, K * K * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed cuI: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&KK, K * K * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed KK: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&KK2, K * K * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed KK2: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&KVOXELS, K * Globals::colsX * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed KVOXELS: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&KVOXELS2, K * Globals::colsX * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed KVOXELS: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&TIMECOMPK, Globals::rowsX * K * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed TIMECOMPK: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&TIMECOMPK2, Globals::rowsX * K * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed TIMECOMPK2: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&Kv, K * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed Kv: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void **)&TIMECOMPVOXELS, Globals::rowsX * Globals::colsX * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed TIMECOMPVOXELS: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void**)&d_W, K * sizeof(datatype))) != cudaSuccess) {
		printf("cudaMalloc failed d_W: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
	if ((err = cudaMalloc((void**)&devInfo, sizeof(int))) != cudaSuccess) {
		printf("cudaMalloc failed devInfo: %s\n", cudaGetErrorString(err));
		cleanup();
		exit(-1);
	}
}