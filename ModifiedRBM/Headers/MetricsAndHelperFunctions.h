#ifndef _METRIX_AND_HELPER_FUNCTIONS_H_
#define _METRIX_AND_HELPER_FUNCTIONS_H_

#include "DataType.h"
#include "ComplexMKL.h"
#include "CRSMatrix.h"

MKL_Complex16* GetRandomDiagMatrix(int N);
double NormMatrixDiag(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* RoMatrixRBM);
double MaxEigDiffMatrix(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* RoMatrixRBM);
double KullbachLeiblerNorm(int N, MKL_Complex16** OriginalRoMatrices, MKL_Complex16* RoMatrixRBM, int NumberOfBases, CRSMatrix* UbMatrices);
double KullbachLeiblerNorm(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* NewRoMatrixRBM);
double GetFidelity(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* RoMatrixRBM);
void GetUnitaryMatrix(int seed, int N, MKL_Complex16* A);
CRSMatrix* GetUbRandomMatrices(int N, int NumberOfBases, bool check = false);

#endif //_METRIX_AND_HELPER_FUNCTIONS_H_