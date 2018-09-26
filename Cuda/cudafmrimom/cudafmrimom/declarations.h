///////////////////////////////////////////////
//         File: declarations.h              //
///////////////////////////////////////////////
#ifndef __declarations__
#define __declarations__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MODE 1	// 0 for float, 1 for double

#if (MODE)
	typedef double datatype;
#else
	typedef float datatype;
#endif

#define K 10 // The number of sources
#define ITER 200 // The total number of iterations
#define TS 20 // The number of internal iterations of NewCoef
#define TD 20 // The number of internal iterations of NewDict
#define LAMBDA 0.5 // The ë of the problem (depends on the dataset)
#define CCL 1 // Value of the normalization of each atom 
#define ES -1 //(optional) If it is selected, the stop criteria of the spatial maps is changed to the error Es between iterations given by ||S^[n]-S^[n-1]||_{F} < Es 
#define ED -1 //(optional) The stop criteria is changed to use an  error check step ||D^[n]-D^[n-1]||_{F} < Ed

/* In order to use read mat files */
#define	Xarray_in_PATH		"./input/X.mat"
#define	MATLAB_include		C:\Program Files\MATLAB\MATLAB Production Server\R2015a\extern\include\\

#define QUOTEME(x)			QUOTEME_1(x)
#define QUOTEME_1(x)		#x
#define INCLUDE_F(x,y)		QUOTEME(x ## y)
#define INCLUDE_FILE(x,y)	INCLUDE_F(x,y)

#endif