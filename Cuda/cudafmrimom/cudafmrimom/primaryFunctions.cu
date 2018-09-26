///////////////////////////////////////////////
//       File: primaryFunctions.cu           //
///////////////////////////////////////////////
#include <iostream>
#include <curand.h>
#include <cusolverDn.h>
#include "declarations.h"
#include "cublas_v2.h"
#include "kernel.h"
#include "cuLibs.h"
#include "reduction.h"
#include "errorTypes.h"
#include "Globals.h"
#include "mem.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////
/* Global variables */
extern int *devInfo;
extern cublasHandle_t handle;
extern curandGenerator_t generator;
extern cusolverDnHandle_t cusolverH;
extern datatype *I, *KK, *KK2, *KVOXELS, *KVOXELS2, *TIMECOMPK, *TIMECOMPK2, *Kv, *TIMECOMPVOXELS, *X, *D, *S, *W, *d_W;
///////////////////////////////////////////////////////////////////////////////


/* Computes the error of the approximation */
datatype ErrorComp() {
	gpu_blas_mmul(handle, D, S, TIMECOMPVOXELS, 0, Globals::rowsX, K, K, Globals::colsX); // D*S
	subtractMatrices << <(Globals::rowsX*Globals::colsX) / 1024 + 1, 1024 >> > (TIMECOMPVOXELS, X, TIMECOMPVOXELS, Globals::rowsX*Globals::colsX); // Y - D*S			
	return sqrt(frobeniusNorm(handle, TIMECOMPVOXELS, Globals::rowsX*Globals::colsX) / (Globals::rowsX*Globals::colsX)); // sqrt(norm(Y - D*S, 'fro') / (TIMECOMP*VOXELS))
}


/* Calculates the new coefficients according to the MM algorithm */
void NewCoef(datatype lamb) {
	cudaError_t err;
	datatype cS, Err, bound;
	int t, Ts, Es;
	if (ES != -1) {
		Ts = 500;
		Es = ES;
	}
	else {
		Ts = TS;
		Es = 0;
	}
	gpu_blas_mmul(handle, D, D, KK, 1, Globals::rowsX, K, Globals::rowsX, K); // Dux = D'*D;
	gpu_blas_mmul(handle, KK, KK, KK2, 1, K, K, K, K); // Dux.'*Dux
	cS = findMaxSqrtEigenvalue(cusolverH, KK2, W, d_W, devInfo, K); // cS  = max(sqrt(eig(Dux.'*Dux)))
	gpu_blas_mmul(handle, D, X, KVOXELS, 1, Globals::rowsX, K, Globals::rowsX, Globals::colsX); // D'*Y
	matrixDivNum << <(K*Globals::colsX) / 1024 + 1, 1024 >> > (KVOXELS, K*Globals::colsX, cS); // DY = D'*Y/cS
	matrixDivNum << < (K*K) / 1024 + 1, 1024 >> > (KK, K*K, cS); // Dux/cS
	subtractMatrices << <(K*K) / 1024 + 1, 1024 >> > (KK2, I, KK, K*K); //Aq = I - Dux / cS;
	Err = 1.0;
	t = 1;
	bound = (datatype)0.5*lamb / cS;
	while (t <= Ts && Err>Es) {
		gpu_blas_mmul(handle, KK2, S, KVOXELS2, 0, K, K, K, Globals::colsX); // Aq*S
		addMatrices << <(K*Globals::colsX) / 1024 + 1, 1024 >> > (KVOXELS2, KVOXELS, KVOXELS2, K*Globals::colsX); // A = DY+Aq*S
		wthresh << <(K*Globals::colsX) / 1024 + 1, 1024 >> > (KVOXELS2, K*Globals::colsX, bound); // A = wthresh(A, 's', 0.5*lam / cS)
		if ((err = cudaMemcpy(S, KVOXELS2, K * Globals::colsX * sizeof(datatype), cudaMemcpyDeviceToDevice)) != cudaSuccess) {
			printf("cudaMemcpy S failed: %s\n", cudaGetErrorString(err));
			cleanup();
			exit(-1);
		}	// S=A
		t = t + 1;
	}
}


/* Calculates the new Dictionary atoms according to the MM algorithm */
void NewDict() {
	int t, Ed, Td;
	cudaError_t err;
	datatype cD, Err;
	if (ED != -1) {
		Ed = ED;
		Td = 100;
	}
	else {
		Td = TD;
		Ed = 0;
	}
	gpu_blas_mmul(handle, S, S, KK, 2, K, Globals::colsX, K, Globals::colsX); // Sux = S*S'
	gpu_blas_mmul(handle, KK, KK, KK2, 1, K, K, K, K); // Sux.'*Sux
	cD = findMaxSqrtEigenvalue(cusolverH, KK2, W, d_W, devInfo, K); //  cD  = max(sqrt(eig(Sux.'*Sux)))
	gpu_blas_mmul(handle, X, S, TIMECOMPK, 2, Globals::rowsX, Globals::colsX, K, Globals::colsX); // Y*S'
	matrixDivNum << <(Globals::rowsX*K) / 1024 + 1, 1024 >> > (TIMECOMPK, Globals::rowsX*K, cD); // YS = Y*S'/cD
	matrixDivNum << <(K*K) / 1024 + 1, 1024 >> > (KK, K*K, cD); // Sux/cD
	subtractMatrices << <(K*K) / 1024 + 1, 1024 >> > (KK2, I, KK, K*K); // Bq = I-Sux/cD
	Err = 1.0;
	t = 1;
	while (t <= Td && Err > Ed) {
		gpu_blas_mmul(handle, D, KK2, TIMECOMPK2, 0, Globals::rowsX, K, K, K); // D*Bq
		addMatrices << <(Globals::rowsX*K) / 1024 + 1, 1024 >> > (TIMECOMPK2, TIMECOMPK, TIMECOMPK2, Globals::rowsX*K); // B = YS + D*Bq
		normalizeReduction(TIMECOMPK2, Kv); // Kv = sqrt(sum(B.^2)); Kv(Kv < ccl) = 1; 
		rdivide << <(Globals::rowsX*K) / 1024 + 1, 1024 >> > (TIMECOMPK2, Globals::rowsX, K, Kv); //B = bsxfun(@rdivide,B,Kv)
		if ((err = cudaMemcpy(D, TIMECOMPK2, Globals::rowsX * K * sizeof(datatype), cudaMemcpyDeviceToDevice)) != cudaSuccess) {
			printf("cudaMemcpy failed D: %s\n", cudaGetErrorString(err));
			cleanup();
			exit(-1);
		}  //D = B
		t = t + 1;
	}
}


/* Majorized minimization */
void mom() {
	int i;
	datatype lamb;
	lamb = (datatype)LAMBDA*sqrt(frobeniusNorm(handle, X, Globals::rowsX*Globals::colsX) / (Globals::rowsX*Globals::colsX)); //lamb = lamb*sqrt(norm(Y,'fro')/(T*N));		
	randn(generator, D, Globals::rowsX * K); // D = randn(T,K)
	randn(generator, S, K * Globals::colsX); // S = randn(K,N)
	dim3 grid(1, 1);
	dim3 threads(K, K);
	initIdentityGPU << <grid, threads >> > (I, K, K);
	printf("Initial error\t:  %f\n\n", ErrorComp());
	for (i = 0; i < ITER; i++) {
		NewCoef(lamb);
		NewDict();
		printf("Iteration %d\t:  %f \n", i + 1, ErrorComp());
	}
	printf("\nFinal error\t:  %f \n", ErrorComp());
}