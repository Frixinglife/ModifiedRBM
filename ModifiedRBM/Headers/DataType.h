#ifndef _DATA_TYPE_H_
#define _DATA_TYPE_H_

#define ACC_FLOAT

#ifdef ACC_DOUBLE
	#define TComplex MKL_Complex16
	#define acc_number double
	#define TRngUniform vdRngUniform
	#define Tcblas_v cblas_dgemv
	#define MATRIX_OUT "..\\Results\\matrix_diag_double.txt"
	#define TYPE_OUT "double"
	#define EPS (double)1e-12
#else
#ifdef ACC_FLOAT
	#define TComplex MKL_Complex8
	#define acc_number float
	#define TRngUniform vsRngUniform
	#define Tcblas_v cblas_sgemv
	#define MATRIX_OUT "..\\Results\\matrix_diag_float.txt"
	#define TYPE_OUT "float"
	#define EPS (float)1e-8
#endif
#endif

#endif //_DATA_TYPE_H_
