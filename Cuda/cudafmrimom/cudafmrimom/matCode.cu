///////////////////////////////////////////////
//            File: matCode.cu               //
///////////////////////////////////////////////
#include <iostream>
#include "mem.h"
#include "Globals.h"
#include "declarations.h"
#include INCLUDE_FILE(MATLAB_include,mat.h)
#include INCLUDE_FILE(MATLAB_include,matrix.h)


using namespace std;


/* Read MATLAB file format */
bool readMat(datatype** array_dbl, mxArray **mArray, unsigned int* rows_ptr, unsigned int* cols_ptr, char* file, char* array_name) {
	MATFile *pmat;
	if ((pmat = matOpen(file, "r") ) == NULL) {
		cout << "Error opening file " << file << " ..." << endl;
		return false;
	}
	if ((*mArray = matGetVariable(pmat, array_name)) == NULL) {
		cerr << "Error reading existing matrix X..." << endl;
		matClose(pmat);
		return false;
	}
	#ifdef MODE
		*array_dbl = mxGetPr(*mArray);
	#else
		*array_dbl = (datatype*)mxGetData(*mArray);
	#endif
	*rows_ptr = (unsigned int)mxGetM(*mArray);
	*cols_ptr = (unsigned int)mxGetN(*mArray);
	if (matClose(pmat) != 0) {
		cerr << "Error closing file " << file << "..." << endl;
		return false;
	}
	return true;
}


/* Uses readMat function to read input from mat file and stores it to given array */
void readingInput(datatype **array, mxArray** mx) {
	if (!readMat(array, mx, &Globals::rowsX, &Globals::colsX, Xarray_in_PATH, "X")) {
		printf("An error occured while reading input file!");
		cleanup();
		exit(-1);
	}
}