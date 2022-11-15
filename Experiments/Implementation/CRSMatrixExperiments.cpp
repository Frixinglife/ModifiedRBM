#include "TransitionMatrix.h"
#include "CRSMatrix.h"
#include <iostream>

void CRSMatrixTest() {
    int n = 3;

    MKL_Complex16 One(1.0, 0.0), Zero(0.0, 0.0), I(0.0, 1.0);

    MKL_Complex16 mass[9] = {
        One, Zero, I,
        I, I, Zero,
        Zero, One, Zero
    };

    CRSMatrix A(n, mass);
    A.PrintCRS("A");

    CRSMatrix B = A.GetHermitianConjugateCRS();
    B.PrintCRS("B");
}

void CRSMatrixMultTest() {
    int n = 2;

    MKL_Complex16 One(1.0, 0.0), Zero(0.0, 0.0);

    MKL_Complex16 mass[4] = {
        One, Zero,
        One, Zero
    };

    CRSMatrix A(n, mass);
    A.PrintCRS("A");

    MKL_Complex16 B[4] = {
        Zero, One,
        One, Zero
    };

    TransitionMatrix::PrintMatrix(B, n, n, "B");

    MKL_Complex16* AB = CRSMatrix::MultCRSDense(A, B, n);
    MKL_Complex16* BA = CRSMatrix::MultDenseCSR(B, A, n);

    TransitionMatrix::PrintMatrix(AB, n, n, "A * B");
    TransitionMatrix::PrintMatrix(BA, n, n, "B * A");

    delete[] AB;
    delete[] BA;
}
