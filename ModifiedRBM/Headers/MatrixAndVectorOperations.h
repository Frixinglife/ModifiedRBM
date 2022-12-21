#ifndef _MATRIX_AND_VECTOR_OPERATIONS_H_
#define _MATRIX_AND_VECTOR_OPERATIONS_H_

#include "ComplexMKL.h"
#include "DataType.h"

class MatrixAndVectorOperations {
public:
    static void VectorsAdd(int N, acc_number* FirstVec, acc_number* SecondVec, acc_number* Result);
    static void VectorsSub(int N, acc_number* FirstVec, acc_number* SecondVec, acc_number* Result);
    static void VectorsAdd(int N, TComplex* FirstVec, TComplex* SecondVec, TComplex* Result);
    static void VectorsSub(int N, TComplex* FirstVec, TComplex* SecondVec, TComplex* Result);
    static acc_number ScalarVectorMult(int N, acc_number* FirstVec, acc_number* SecondVec);
    static void MultVectorByNumber(int N, acc_number* Vec, acc_number Number, acc_number* Result);
    static void MultVectorByNumberInPlace(int N, acc_number* Vec, acc_number Number);
    static void MultVectorByNumberInPlace(int N, TComplex* Vec, TComplex Number);
    static void MultMatrixByNumberInPlace(int N, int M, acc_number* Matrix, acc_number Number);
    static void MatrixAdd(int N, int M, acc_number* FirstMatrix, acc_number* SecondMatrix, acc_number* Result);
    static void MatrixSub(int N, int M, acc_number* FirstMatrix, acc_number* SecondMatrix, acc_number* Result);
    static void MultMatrixByNumberInPlace(int N, int M, TComplex* Matrix, TComplex Number);
    static void MatrixAdd(int N, int M, TComplex* FirstMatrix, TComplex* SecondMatrix, TComplex* Result);
    static void MatrixSub(int N, int M, TComplex* FirstMatrix, TComplex* SecondMatrix, TComplex* Result);
    static void MatrixVectorMult(int N, int M, acc_number* Matrix, acc_number* Vec, acc_number* Result);
    static void VectorVectorMult(int N, int M, TComplex* FirstVec, TComplex* SecondVec, TComplex* MatrixResult);
    static void VectorVectorMult(int N, int M, acc_number* FirstVec, acc_number* SecondVec, acc_number* MatrixResult);
    static void FindEigMatrix(int N, MKL_Complex16* Matrix, MKL_Complex16* Result);
    static void SqrtMatrix(int N, MKL_Complex16* Matrix, MKL_Complex16* Result);
};

#endif //_MATRIX_AND_VECTOR_OPERATIONS_H_
