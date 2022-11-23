#ifndef _TRANSITION_MATRIX_H_
#define _TRANSITION_MATRIX_H_

#include "ComplexMKL.h"
#include "DataType.h"
#include "CRSMatrix.h"

class TransitionMatrix {
private:
    VSLStreamStatePtr stream;

public:
    TransitionMatrix(int seed = 42);
    ~TransitionMatrix();

    static void PrintMatrix(MKL_Complex16* Matrix, int n, int m, std::string name);
    void GetUnitaryMatrices(MKL_Complex16* Matrices, int NumberOfU, int NumberOfUnitary = 1, int IndexUnitary = 0,
        double left = 0.0, double right = 1.0);
    void GetUnitaryMatrix(MKL_Complex16* Matrix, double a, double b, double c, double d);
    void GetIdentityMatrix(MKL_Complex16* Matrix);
    static MKL_Complex16 ComplexMult(MKL_Complex16 A, MKL_Complex16 B);
    void ShowUnitaryMatrices(MKL_Complex16* Matrices, int NumberOfU);
    void KroneckerProduction(MKL_Complex16* Matrix_A, int size_A, MKL_Complex16* Matrix_B, int size_B,
        MKL_Complex16* Matrix_Res, bool show = false);

    MKL_Complex16* GetTransitionMatrix(int N, int NumberOfUnitary = 1, int IndexUnitary = 0, bool show = false);
    static MKL_Complex16* GetHermitianConjugateMatrix(MKL_Complex16* Matrix, int N);
    static MKL_Complex16* GetNewRoMatrix(MKL_Complex16* Ro, MKL_Complex16* Ub, MKL_Complex16* Ub_t, int N);

    CRSMatrix GetCRSTransitionMatrix(int N, int NumberOfUnitary = 1, int IndexUnitary = 0, bool show = false);
    static MKL_Complex16* GetNewRoMatrix(MKL_Complex16* Ro, CRSMatrix& Ub, int N);
};

#endif //_TRANSITION_MATRIX_H_
