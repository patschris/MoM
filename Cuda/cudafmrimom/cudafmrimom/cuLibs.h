///////////////////////////////////////////////
//            File: cuLibs.h                 //
///////////////////////////////////////////////
#ifndef __cuLibs__
#define __cuLibs__

datatype frobeniusNorm(cublasHandle_t, datatype *, int);
void randn(curandGenerator_t, datatype *, int);
datatype findMaxSqrtEigenvalue(cusolverDnHandle_t, datatype *, datatype *, datatype *, int *, const int);
void gpu_blas_mmul(cublasHandle_t, const datatype *, const datatype *, datatype *, int, const int, const int, const int, const int);

#endif