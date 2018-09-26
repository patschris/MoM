///////////////////////////////////////////////
//             File: kernel.h                //
///////////////////////////////////////////////
#ifndef __kernel__
#define __kernel__

__global__ void initIdentityGPU(datatype *, int, int);
__global__ void matrixDivNum(datatype *, int, datatype);
__global__ void subtractMatrices(datatype *, datatype *, datatype *, int);
__global__ void addMatrices(datatype *, datatype *, datatype *, int);
__global__ void wthresh(datatype *, int, datatype);
__global__ void rdivide(datatype *, int, int, datatype *);

#endif // !__kernel__