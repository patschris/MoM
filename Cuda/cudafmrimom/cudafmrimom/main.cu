/*********************************************************************/
/*  This function is designed to solve the optimization task of the  */
/*  Assisted Dictionary Learning via Majorization Method also known  */
/*  as the Majorized Minimization algorithm                          */
/*  Based on https://github.com/MorCTI/Attom-Assisted-DL/find/master */
/*  Written in CUDA C using Microsoft Visual Studio 2015             */
/*  Implemented by C. Patsouras, D. Papageorgiou                     */
/*********************************************************************/


///////////////////////////////////////////////
//             File: main.cu                 //
///////////////////////////////////////////////
#include <iostream>
#include <ctime>
#include <windows.h>
#include <curand.h>
#include <cusolverDn.h>
#include "cublas_v2.h"
#include "errorTypes.h"
#include "declarations.h"
#include "Globals.h"
#include "timer.h"
#include "mem.h"
#include INCLUDE_FILE(MATLAB_include,mat.h)
#include INCLUDE_FILE(MATLAB_include,matrix.h)
#include "matCode.h"
#include "primaryFunctions.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////
/* Global variables */
int *devInfo;
cublasHandle_t handle;
curandGenerator_t generator;
cusolverDnHandle_t cusolverH;
datatype *I, *KK, *KK2, *KVOXELS, *KVOXELS2, *TIMECOMPK, *TIMECOMPK2, *Kv, *TIMECOMPVOXELS, *X, *D, *S, *W, *d_W;
///////////////////////////////////////////////////////////////////////////////


int main(void) {
	cudaError_t err;
	datatype *dataInput;
	mxArray *mx_ptr = NULL;

	printf("MoM algorithm\n");

	/* Reading input from file */
	printf("Reading input from file\nPlease wait ...\n");
	{
		Timer t;
		if (!t.isValid()) {
			printf("Timer initialization error! Aborting...\n");
			return -1;
		}
		t.tic();
		readingInput(&dataInput, &mx_ptr);
		printf("Reading Time: %f sec\n", t.toc());
		printf("Input size : %d x %d\n", Globals::rowsX, Globals::colsX);
	}
	/* Allocate all the necessary memory */
	{
		Timer t;
		if (!t.isValid()) {
			cerr << "Timer initialization error! Aborting..." << endl;
			return -1;
		}
		t.GPUtic();
		allocate();
		printf("Time to allocate all the necessary memory: %f sec\n", t.GPUtoc() / 1000.0);
	}
	if ((err = cudaMemcpy(X, dataInput, Globals::rowsX * Globals::colsX * sizeof(datatype), cudaMemcpyHostToDevice)) != cudaSuccess) {
		printf("cudaMemcpy X failed: %s\n", cudaGetErrorString(err));
		cleanup();
		return -1;
	}
	/* Calling MoM */
	{
		Timer t;
		if (!t.isValid()) {
			cerr << "Timer initialization error! Aborting..." << endl;
			return -1;
		}
		t.GPUtic();
		mom();
		printf("MoM Time: %f sec\n", t.GPUtoc() / 1000.0);
	}
	/* Check for possible errors */
	while ((err = cudaGetLastError()) != cudaSuccess) 
		printf("Error: An error occured during the execution of the program: %s\n", cudaGetErrorString(err));
	/* Free allocated memory */
	if (mx_ptr) mxDestroyArray(mx_ptr);
	cleanup();
	return 0;
}