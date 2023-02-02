#include "MatrixAndVectorOperations.h"
#include <cmath>

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

void MatrixAndVectorOperations::VectorsAdd(int N, TComplex* FirstVec, TComplex* SecondVec, TComplex* Result) {
    for (int i = 0; i < N; i++) {
        Result[i] = FirstVec[i] + SecondVec[i];
    }
}

void MatrixAndVectorOperations::VectorsSub(int N, TComplex* FirstVec, TComplex* SecondVec, TComplex* Result) {
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

void MatrixAndVectorOperations::MultVectorByNumberInPlace(int N, acc_number* Vec, acc_number Number) {
    for (int i = 0; i < N; i++) {
        Vec[i] *= Number;
    }
}

void MatrixAndVectorOperations::MultVectorByNumberInPlace(int N, TComplex* Vec, TComplex Number) {
    for (int i = 0; i < N; i++) {
        Vec[i] *= Number;
    }
}

void MatrixAndVectorOperations::MultMatrixByNumberInPlace(int N, int M, acc_number* Matrix, acc_number Number) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            Matrix[j + i * M] *= Number;
        }
    }
}

void MatrixAndVectorOperations::MultMatrixByNumberInPlace(int N, int M, TComplex* Matrix, TComplex Number) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            Matrix[j + i * M] *= Number;
        }
    }
}

void MatrixAndVectorOperations::MatrixAdd(int N, int M, acc_number* FirstMatrix, acc_number* SecondMatrix, acc_number* Result) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            Result[j + i * M] = FirstMatrix[j + i * M] + SecondMatrix[j + i * M];
        }
    }
}

void MatrixAndVectorOperations::MatrixSub(int N, int M, acc_number* FirstMatrix, acc_number* SecondMatrix, acc_number* Result) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            Result[j + i * M] = FirstMatrix[j + i * M] - SecondMatrix[j + i * M];
        }
    }
}

void MatrixAndVectorOperations::MatrixAdd(int N, int M, TComplex* FirstMatrix, TComplex* SecondMatrix, TComplex* Result) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            Result[j + i * M] = FirstMatrix[j + i * M] + SecondMatrix[j + i * M];
        }
    }
}

void MatrixAndVectorOperations::MatrixSub(int N, int M, TComplex* FirstMatrix, TComplex* SecondMatrix, TComplex* Result) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            Result[j + i * M] = FirstMatrix[j + i * M] - SecondMatrix[j + i * M];
        }
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

void MatrixAndVectorOperations::VectorVectorMult(int N, int M, acc_number* FirstVec, acc_number* SecondVec, acc_number* MatrixResult) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            MatrixResult[j + i * M] = FirstVec[i] * SecondVec[j];
        }
    }
}

void MatrixAndVectorOperations::VectorVectorMult(int N, int M, TComplex* FirstVec, TComplex* SecondVec, TComplex* MatrixResult) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            MatrixResult[j + i * M] = FirstVec[i] * SecondVec[j];
        }
    }
}

void MatrixAndVectorOperations::FindEigMatrix(int N, MKL_Complex16* Matrix, MKL_Complex16* Result) {
    const char jobvl = 'N';
    const char jobvr = 'N';

    const int N_N = N * N;
    const int N_2 = 2 * N;

    MKL_Complex16* A = new MKL_Complex16[N_N];
    MKL_Complex16* W = new MKL_Complex16[N];
    MKL_Complex16* VL = new MKL_Complex16[N_N];
    MKL_Complex16* VR = new MKL_Complex16[N_N];
    MKL_Complex16* Work = new MKL_Complex16[N_2];
    double* rwork = new double[N_2];

    for (int i = 0; i < N_N; i++) {
        A[i] = Matrix[i];
    }

    const int lda = N;
    const int ldvl = N;
    const int ldvr = N;
    const int lwork = 2 * N;
    int info;

    zgeev(&jobvl, &jobvr, &N, A, &lda, W, VL, &ldvl, VR, &ldvr, Work, &lwork, rwork, &info);

    for (int i = 0; i < N; i++) {
        Result[i] = W[i];
    }

    delete[]A;
    delete[]W;
    delete[]VL;
    delete[]VR;
    delete[]Work;
    delete[]rwork;
}

void MatrixAndVectorOperations::GetInvMatrix(int N, MKL_Complex16* Matrix, MKL_Complex16* Result) {
    const int N_N = N * N;
    const int N_2 = 2 * N;
    const int lwork = 2 * N;
    const int lda = N;
    int info;

    MKL_Complex16* Work = new MKL_Complex16[N_2];
    MKL_Complex16* LU = new MKL_Complex16[N_N];
    int* ipiv = new int[N];
    for (int i = 0; i < N_N; i++) {
        LU[i] = Matrix[i];
    }
    zgetrf(&N, &N, LU, &lda, ipiv, &info);
    zgetri(&N, LU, &lda, ipiv, Work, &lwork, &info);
    for (int i = 0; i < N_N; i++) {
        Result[i] = LU[i];
    }
}

void MatrixAndVectorOperations::SqrtMatrix(int N, MKL_Complex16* Matrix, MKL_Complex16* Result) {
    const char jobvl = 'N';
    const char jobvr = 'V';
    const int N_N = N * N;
    const int N_2 = 2 * N;
    const int lda = N;
    const int ldvl = N;
    const int ldvr = N;
    const int lwork = 2 * N;
    int info;

    MKL_Complex16* J = new MKL_Complex16[N_N];
    MKL_Complex16* W = new MKL_Complex16[N];
    MKL_Complex16* VL = new MKL_Complex16[N_N];
    MKL_Complex16* VR = new MKL_Complex16[N_N];
    MKL_Complex16* VR_T = new MKL_Complex16[N_N];
    MKL_Complex16* Work = new MKL_Complex16[N_2];
    MKL_Complex16* Intermed = new MKL_Complex16[N_N];
    double* rwork = new double[N_2];

    for (int i = 0; i < N_N; i++) {
        J[i] = Matrix[i];
    }

    zgeev(&jobvl, &jobvr, &N, J, &lda, W, VL, &ldvl, VR, &ldvr, Work, &lwork, rwork, &info);

    for (int i = 0; i < N; i++) {
        double real = J[i + i * N].real();
        double imag = J[i + i * N].imag();
        double sqrt_real = 0.0;
        if (real > 0.0) {
            sqrt_real = std::sqrt(real);
        }
        J[i + i * N] = MKL_Complex16(sqrt_real, imag);
    }

    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            MKL_Complex16 Temp = MKL_Complex16(VR[j + i * N].real(), -VR[j + i * N].imag());
            VR[j + i * N] = MKL_Complex16(VR[i + j * N].real(), -VR[i + j * N].imag());
            VR[i + j * N] = Temp;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            VR_T[j + i * N] = MKL_Complex16(VR[i + j * N].real(), -VR[i + j * N].imag());
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Intermed[j + i * N] = MKL_Complex16(0.0, 0.0);
            for (int k = 0; k < N; k++) {
                Intermed[j + i * N] += VR[k + i * N] * J[j + k * N];
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Result[j + i * N] = MKL_Complex16(0.0, 0.0);
            for (int k = 0; k < N; k++) {
                Result[j + i * N] += Intermed[k + i * N] * VR_T[j + k * N];
            }
        }
    }

    delete[]J;
    delete[]W;
    delete[]VL;
    delete[]VR;
    delete[]Work;
    delete[]rwork;
    delete[]Intermed;
}

//void MatrixAndVectorOperations::SqrtMatrix(int N, MKL_Complex16* Matrix, MKL_Complex16* Result) {
//    Matrix[0] = MKL_Complex16(1.0, 0.0);
//    Matrix[1] = MKL_Complex16(2.0, 0.0);
//    Matrix[2] = MKL_Complex16(3.0, 0.0);
//    Matrix[3] = MKL_Complex16(4.0, 0.0);
//
//    Matrix[4] = MKL_Complex16(0.0, 0.0);
//    Matrix[5] = MKL_Complex16(5.0, 0.0);
//    Matrix[6] = MKL_Complex16(6.0, 0.0);
//    Matrix[7] = MKL_Complex16(7.0, 0.0);
//
//    Matrix[8] = MKL_Complex16(0.0, 0.0);
//    Matrix[9] = MKL_Complex16(0.0, 0.0);
//    Matrix[10] = MKL_Complex16(8.0, 0.0);
//    Matrix[11] = MKL_Complex16(9.0, 0.0);
//
//    Matrix[12] = MKL_Complex16(0.0, 0.0);
//    Matrix[13] = MKL_Complex16(0.0, 0.0);
//    Matrix[14] = MKL_Complex16(0.0, 0.0);
//    Matrix[15] = MKL_Complex16(10.0, 0.0);
//
//    MKL_Z_SELECT_FUNCTION_1 select = NULL;
//    const char jobvs = 'V';
//    const char sort = 'N';
//    const int N_N = N * N;
//    const int N_2 = 2 * N;
//    const int lda = N;
//    const int ldvs = N;
//    const int lwork = 2 * N;
//    int sdim, info, bwork;
//
//    MKL_Complex16* A = new MKL_Complex16[N_N];
//    MKL_Complex16* W = new MKL_Complex16[N];
//    MKL_Complex16* VS = new MKL_Complex16[N_N];
//    MKL_Complex16* VS_INV = new MKL_Complex16[N_N];
//    MKL_Complex16* Intermed = new MKL_Complex16[N_N];
//    MKL_Complex16* Work = new MKL_Complex16[N_2];
//    double* rwork = new double[N];
//
//    for (int i = 0; i < N_N; i++) {
//        A[i] = Matrix[i];
//    }
//
//    //zgees(&jobvs, &sort, select, &N, A, &lda, &sdim, W, VS, &ldvs, Work, &lwork, rwork, &bwork, &info);
//
//    //for (int i = 0; i < N; i++) {
//    //    for (int j = i; j < N; j++) {
//    //        MKL_Complex16 Temp = A[j + i * N];
//    //        A[j + i * N] = A[i + j * N];
//    //        A[i + j * N] = Temp;
//    //    }
//    //}
//
//    for (int i = 0; i < N; i++) {
//        double sqrt_real = 0.0;
//        if (A[i + i * N].real() > 0.0) {
//            sqrt_real = std::sqrt(A[i + i * N].real());
//        }
//        A[i + i * N] = MKL_Complex16(sqrt_real, 0.0);
//    }
//
//    for (int i = 0; i < N; i++) {
//        for (int j = i + 1; j < N; j++) {
//            double sum = A[i + i * N].real() + A[j + j * N].real();
//            if (std::abs(sum) < 1e-6) {
//                A[j + i * N] = MKL_Complex16(0.0, 0.0);
//            } else {
//                for (int k = i + 1; k < j; k++) {
//                    A[j + i * N] -= A[k + i * N] * A[j + k * N];
//                }
//                A[j + i * N] /= sum;
//            }  
//        }
//    }
//
//    /*for (int i = 0; i < N; i++) {
//        for (int j = i; j < N; j++) {
//            MKL_Complex16 Temp = A[j + i * N];
//            A[j + i * N] = A[i + j * N];
//            A[i + j * N] = Temp;
//        }
//    }
//
//    for (int i = 0; i < N; i++) {
//        for (int j = i; j < N; j++) {
//            MKL_Complex16 Temp = MKL_Complex16(VS[j + i * N].real(), -VS[j + i * N].imag());
//            VS[j + i * N] = MKL_Complex16(VS[i + j * N].real(), -VS[i + j * N].imag());
//            VS[i + j * N] = Temp;
//        }
//    }
//
//    GetInvMatrix(N, VS, VS_INV);
//
//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            Intermed[j + i * N] = MKL_Complex16(0.0, 0.0);
//            for (int k = 0; k < N; k++) {
//                Intermed[j + i * N] += VS[k + i * N] * A[j + k * N];
//            }
//        }
//    }
//
//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            Result[j + i * N] = MKL_Complex16(0.0, 0.0);
//            for (int k = 0; k < N; k++) {
//                Result[j + i * N] += Intermed[k + i * N] * VS_INV[j + k * N];
//            }
//        }
//    }*/
//
//    delete[]A;
//    delete[]W;
//    delete[]VS;
//    delete[]Work;
//    delete[]rwork;
//    delete[]VS_INV;
//    delete[]Intermed;
//}
