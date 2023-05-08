#ifndef _DATA_TYPE_H_
#define _DATA_TYPE_H_

#define ACC_FLOAT

#ifdef ACC_DOUBLE
	#define TComplex MKL_Complex16
	#define acc_number double
	#define TRngUniform vdRngUniform
	#define TRngGaussian vdRngGaussian
	#define Tcblas_v cblas_dgemv
	#define MATRIX_OUT "..\\Results\\double\\matrix_diag.txt"
	#define TYPE_OUT "double"
#else
#ifdef ACC_FLOAT
	#define TComplex MKL_Complex8
	#define acc_number float
	#define TRngUniform vsRngUniform
	#define TRngGaussian vsRngGaussian
	#define Tcblas_v cblas_sgemv
	#define MATRIX_OUT "..\\Results\\float\\matrix_diag.txt"
	#define TYPE_OUT "float"
#endif
#endif

#endif //_DATA_TYPE_H_
