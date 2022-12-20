#include "TransitionMatrixCRSExperiments.h"
#include "TransitionMatrixExperiments.h"
#include "CRSMatrixExperiments.h"
#include "RoMatrixExperiments.h"
#include "TrainingExperiments.h"

#include "NeuralDensityOperators.h"
#include <iostream>
#include <iomanip>

void PrintMatrix(MKL_Complex16* Matrix, int n, std::string name) {
    std::cout << name << ":\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setw(30) << Matrix[j + i * n];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void GetInvMatrix(int N, MKL_Complex16* Matrix, MKL_Complex16* Result) {
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

void FindEigMatrix(int N, MKL_Complex16* Matrix) {
    const char jobvl = 'N';
    const char jobvr = 'V';

    const int N_N = N * N;
    const int N_2 = 2 * N;

    MKL_Complex16* J = new MKL_Complex16[N_N];
    MKL_Complex16* W = new MKL_Complex16[N];
    MKL_Complex16* VL = new MKL_Complex16[N_N];
    MKL_Complex16* VR = new MKL_Complex16[N_N];
    MKL_Complex16* Work = new MKL_Complex16[N_2];
    double* rwork = new double[N_2];

    for (int i = 0; i < N_N; i++) {
        J[i] = Matrix[i];
    }
    // Otherwise Matrix will be overwritten

    const int lda = N;
    const int ldvl = N;
    const int ldvr = N;
    const int lwork = 2 * N;
    int info;

    zgeev(&jobvl, &jobvr, &N, J, &lda, W, VL, &ldvl, VR, &ldvr, Work, &lwork, rwork, &info);

    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            MKL_Complex16 Temp = MKL_Complex16(VR[j + i * N].real(), -VR[j + i * N].imag());
            VR[j + i * N] = MKL_Complex16(VR[i + j * N].real(), -VR[i + j * N].imag());
            VR[i + j * N] = Temp;
        }
    }

    PrintMatrix(Matrix, N, "A (original matrix)");
    PrintMatrix(J, N, "J (eigenvalues)");
    PrintMatrix(VR, N, "S (eigenvectors)");

    MKL_Complex16* VR_T = new MKL_Complex16[N_N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            VR_T[j + i * N] = MKL_Complex16(VR[i + j * N].real(), -VR[i + j * N].imag());
        }
    }

    PrintMatrix(VR_T, N, "S^T (eigenvectors)");
    GetInvMatrix(N, VR, VR_T);
    PrintMatrix(VR_T, N, "S^(-1) (eigenvectors)");

    MKL_Complex16* Intermed = new MKL_Complex16[N_N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Intermed[j + i * N] = MKL_Complex16(0.0, 0.0);
            for (int k = 0; k < N; k++) {
                Intermed[j + i * N] += VR[k + i * N] *  J[j + k * N];
            }
        }
    }

    MKL_Complex16* Result = new MKL_Complex16[N_N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Result[j + i * N] = MKL_Complex16(0.0, 0.0);
            for (int k = 0; k < N; k++) {
                Result[j + i * N] += Intermed[k + i * N] * VR_T[j + k * N];
            }
        }
    }

    PrintMatrix(Result, N, "A = S * J * S^(-1)");

    delete[]Intermed;
    delete[]Result;
    delete[]J;

    delete[]W;
    delete[]VL;
    delete[]VR;
    delete[]Work;
    delete[]rwork;
}

void FindEigMatrixZheev(int N, MKL_Complex16* Matrix) {
    const char jobz = 'V';
    const char uplo = 'U';

    const int N_N = N * N;
    const int N_2 = 2 * N;

    MKL_Complex16* S = new MKL_Complex16[N_N];
    MKL_Complex16* J = new MKL_Complex16[N_N];
    double* W = new double[N];
    MKL_Complex16* Work = new MKL_Complex16[N_2];
    double* rwork = new double[N_2];

    for (int i = 0; i < N_N; i++) {
        J[i] = MKL_Complex16(0.0, 0.0);
        S[i] = Matrix[i];
    }

    const int lda = N;
    const int lwork = 2 * N;
    int info;

    zheev(&jobz, &uplo, &N, S, &lda, W, Work, &lwork, rwork, &info);

    for (int i = 0; i < N; i++) {
        J[i + i * N] = MKL_Complex16(W[i], 0.0);
    }

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            MKL_Complex16 Temp = S[j + i * N];
            S[j + i * N] = S[i + j * N];
            S[i + j * N] = Temp;
        }
    }

    PrintMatrix(Matrix, N, "A (original matrix)");
    PrintMatrix(J, N, "J (eigenvalues)");
    PrintMatrix(S, N, "S (eigenvectors)");

    MKL_Complex16* S_T = new MKL_Complex16[N_N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            S_T[j + i * N] = MKL_Complex16(S[i + j * N].real(), -S[i + j * N].imag());
        }
    }
    PrintMatrix(S_T, N, "S^T (eigenvectors)");

    GetInvMatrix(N, S, S_T);

    PrintMatrix(S_T, N, "S^(-1) (eigenvectors)");

    MKL_Complex16* Intermed = new MKL_Complex16[N_N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Intermed[j + i * N] = MKL_Complex16(0.0, 0.0);
            for (int k = 0; k < N; k++) {
                Intermed[j + i * N] += S[k + i * N] * J[j + k * N];
            }
        }
    }

    MKL_Complex16* Result = new MKL_Complex16[N_N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Result[j + i * N] = MKL_Complex16(0.0, 0.0);
            for (int k = 0; k < N; k++) {
                Result[j + i * N] += Intermed[k + i * N] * S_T[j + k * N];
            }
        }
    }

    PrintMatrix(Result, N, "A = S * J * S^(-1)");

    delete[]Intermed;
    delete[]Result;
    delete[]J;
    delete[]S;
    delete[]S_T;

    delete[]W;
    delete[]Work;
    delete[]rwork;
}

int main() {
    //int N_v, N_h, N_a;
    //N_v = 32;
    //N_h = N_a = 4;
    //int NumberOfBases = 5;
    //int NumberOfUnitary = 1;
    //acc_number lr = (acc_number)1e-2;
    //int epochs = 2000;
    //int freq = 2;

    //TrainingExperimentSeparatelyForBases(N_v, N_h, N_a, NumberOfBases, NumberOfUnitary, epochs, lr, freq);

    int N = 4;
    NeuralDensityOperators SupportRBM(N, 1, 1, 777);
    MKL_Complex16* OriginalRoMatrix = SupportRBM.GetRoMatrix();
    FindEigMatrix(N, OriginalRoMatrix);
    //FindEigMatrixZheev(N, OriginalRoMatrix);

    return 0;
}
