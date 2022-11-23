#ifndef _CRS_MATRIX_H_
#define _CRS_MATRIX_H_

#include "ComplexMKL.h"
#include "DataType.h"
#include <string>

class CRSMatrix {
public:
    int n, nz;
    MKL_Complex16* val;
    int* colIndex;
    int* rowPtr;

    CRSMatrix(int _n = 0, int _nz = 0);
    CRSMatrix(int _n, int _nz, MKL_Complex16* _val, int* _colIndex, int* _rowPtr);
    CRSMatrix(int _n, MKL_Complex16* matrix);
    CRSMatrix(const CRSMatrix& matrix);
    CRSMatrix& operator=(const CRSMatrix& matrix);
    ~CRSMatrix();

    CRSMatrix GetHermitianConjugateCRS();
    void PrintCRS(std::string name);

    static MKL_Complex16* MultCRSDense(CRSMatrix CRS_matrix, MKL_Complex16* Dense_matrix, int n);
    static MKL_Complex16* MultDenseCSR(MKL_Complex16* Dense_matrix, CRSMatrix CRS_matrix, int n);
};

#endif //_CRS_MATRIX_H_
