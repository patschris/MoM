///////////////////////////////////////////////
//          File: errorTypes.h              //
///////////////////////////////////////////////
#ifndef __errorTypes__
#define __errorTypes__

const char *curandGetErrorString(curandStatus_t);
const char* cublasGetErrorString(cublasStatus_t);
const char* cusolverGetErrorString(cusolverStatus_t);

#endif // !__errorTypes__