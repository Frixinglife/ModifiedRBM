#include "MatrixAndVectorOperations.h"

void MatrixAndVectorOperations::VectorsAdd(int N, acc_number* FirstVec, acc_number* SecondVec, acc_number* Result) {
    for (int i = 0; i < N; i++) {
        Result[i] = FirstVec[i] + SecondVec[i];
    }
}

void MatrixAndVectorOperations::VectorsSub(int N, acc_number* FirstVec, acc_number* SecondVec, acc_number* Result) {
    for (int i = 0; i < N; i++) {
        Result[i] = FirstVec[i] - SecondVec[i];
    }
}

acc_number MatrixAndVectorOperations::ScalarVectorMult(int N, acc_number* FirstVec, acc_number* SecondVec) {
    acc_number Answer = (acc_number)0.0;

    for (int i = 0; i < N; i++) {
        Answer += FirstVec[i] * SecondVec[i];
    }

    return Answer;
}

void MatrixAndVectorOperations::MultVectorByNumber(int N, acc_number* Vec, acc_number Number, acc_number* Result) {
    for (int i = 0; i < N; i++) {
        Result[i] = Vec[i] * Number;
    }
}

void MatrixAndVectorOperations::MatrixVectorMult(int N, int M, acc_number* Matrix, acc_number* Vec, acc_number* Result) {
    acc_number alpha = (acc_number)1.0;
    acc_number beta = (acc_number)0.0;

    int lda = M;
    int incx = 1;
    int incy = 1;

    Tcblas_v(CblasRowMajor, CblasNoTrans, N, M, alpha, Matrix, lda, Vec, incx, beta, Result, incy);
}

void MatrixAndVectorOperations::FindEigMatrix(int N, MKL_Complex16* Matrix, MKL_Complex16* Result) {
    const char jobvl = 'N';
    const char jobvr = 'N';

    const int N_N = N * N;
    const int N_2 = 2 * N;

    MKL_Complex16* W = new MKL_Complex16[N];
    MKL_Complex16* VL = new MKL_Complex16[N_N];
    MKL_Complex16* VR = new MKL_Complex16[N_N];
    MKL_Complex16* Work = new MKL_Complex16[N_2];
    double* rwork = new double[N_2];

    const int lda = N;
    const int ldvl = N;
    const int ldvr = N;
    const int lwork = 2 * N;
    int info;

    zgeev(&jobvl, &jobvr, &N, Matrix, &lda, W, VL, &ldvl, VR, &ldvr, Work, &lwork, rwork, &info);

    for (int i = 0; i < N; i++) {
        Result[i] = W[i];
    }

    delete[]W;
    delete[]VL;
    delete[]VR;
    delete[]Work;
    delete[]rwork;
}
