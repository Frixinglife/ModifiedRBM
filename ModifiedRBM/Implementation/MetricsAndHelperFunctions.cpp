#include "MatrixAndVectorOperations.h"
#include "MetricsAndHelperFunctions.h"
#include "TransitionMatrix.h"

MKL_Complex16* GetRandomDiagMatrix(int N) {
	MKL_Complex16* Matrix = new MKL_Complex16[N * N];
	for (int i = 0; i < N * N; i++) {
		Matrix[i] = MKL_Complex16(0.0, 0.0);
	}

	acc_number* random_numbers = new acc_number[N];
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MT19937, 10);
	TRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, random_numbers, 0.0, 1.0);
	vslDeleteStream(&stream);

	acc_number sum = (acc_number)0.0;
	for (int i = 0; i < N; i++) {
		sum += random_numbers[i];
	}

	for (int i = 0; i < N; i++) {
		Matrix[i + i * N] = MKL_Complex16(random_numbers[i] / sum, 0.0);
	}
	delete[] random_numbers;

	return Matrix;
}

double NormMatrixDiag(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* RoMatrixRBM) {
    double result = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = OriginalRoMatrix[i + i * N].real() - RoMatrixRBM[i + i * N].real();
        result += diff * diff;
    }
    return std::sqrt(result);
}

double MaxEigDiffMatrix(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* RoMatrixRBM) {
    MKL_Complex16* Diff = new MKL_Complex16[N * N];
    for (int i = 0; i < N * N; i++) {
        Diff[i] = OriginalRoMatrix[i] - RoMatrixRBM[i];
    }

    MKL_Complex16* EigVector = new MKL_Complex16[N];
    MatrixAndVectorOperations::FindEigMatrix(N, Diff, EigVector);

    double MaxLambda = -1e6;

    for (int i = 0; i < N; i++) {
        double Re = EigVector[i].real();
        double Im = EigVector[i].imag();
        double Value = std::sqrt(Re * Re + Im * Im);
        if (Value > MaxLambda) {
            MaxLambda = Value;
        }
    }

    delete[] Diff;
    delete[] EigVector;

    return MaxLambda;
}

double KullbackLeiblerNorm(int N, MKL_Complex16** OriginalRoMatrices, MKL_Complex16* RoMatrixRBM, int NumberOfBases, CRSMatrix* UbMatrices) {
    double result = 0.0;
    for (int b = 0; b < NumberOfBases; b++) {
        MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrixRBM, UbMatrices[b], N);
        for (int i = 0; i < N; i++) {
            double OrigRo_elem = OriginalRoMatrices[b][i + i * N].real();
            double RoRBM_elem = NewRoMatrix[i + i * N].real();
            if (std::abs(OrigRo_elem) >= 1e-5 && std::abs(RoRBM_elem) >= 1e-5) {
                result += OrigRo_elem * std::log(OrigRo_elem / RoRBM_elem);
            }
        }
        delete[] NewRoMatrix;
    }
    return result;
}

double GetFidelity(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* RoMatrixRBM) {
    const int N_N = N * N;
    MKL_Complex16* SqrtRoMatrix = new MKL_Complex16[N_N];
    MatrixAndVectorOperations::SqrtMatrix(N, RoMatrixRBM, SqrtRoMatrix);

    MKL_Complex16* Intermed = new MKL_Complex16[N_N];
    MKL_Complex16* Result = new MKL_Complex16[N_N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Intermed[j + i * N] = MKL_Complex16(0.0, 0.0);
            for (int k = 0; k < N; k++) {
                Intermed[j + i * N] += SqrtRoMatrix[k + i * N] * OriginalRoMatrix[j + k * N];
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Result[j + i * N] = MKL_Complex16(0.0, 0.0);
            for (int k = 0; k < N; k++) {
                Result[j + i * N] += Intermed[k + i * N] * SqrtRoMatrix[j + k * N];
            }
        }
    }

    MatrixAndVectorOperations::SqrtMatrix(N, Result, Intermed);

    double Trace = 0.0;
    for (int i = 0; i < N; i++) {
        Trace += Intermed[i + i * N].real();
    }

    delete[]Result;
    delete[]Intermed;
    delete[]SqrtRoMatrix;

    return Trace;
}

void GetUnitaryMatrix(int seed, int N, MKL_Complex16* A) {
    const int lda = N;
    const int N_N = N * N;
    const int lwork = N;
    int info;
    MKL_Complex16* tau = new MKL_Complex16[N];
    MKL_Complex16* Work = new MKL_Complex16[lwork];
    double* A_double = new double[N_N];

    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N_N, A_double, 0.0, 1.0);
    vslDeleteStream(&stream);

    for (int i = 0; i < N_N; i++) {
        A[i] = MKL_Complex16(A_double[i], 0.0);
    }

    zgeqrfp(&N, &N, A, &lda, tau, Work, &lwork, &info);
    zungqr(&N, &N, &N, A, &lda, tau, Work, &lwork, &info);

    delete[] tau;
    delete[] Work;
    delete[] A_double;
}

double KullbackLeiblerNorm(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* NewRoMatrixRBM) {
    double result = 0.0;
    for (int i = 0; i < N; i++) {
        double OrigRo_elem = OriginalRoMatrix[i + i * N].real();
        double RoRBM_elem = NewRoMatrixRBM[i + i * N].real();
        if (std::abs(OrigRo_elem) >= 1e-5 && std::abs(RoRBM_elem) >= 1e-5) {
            result += OrigRo_elem * std::log(OrigRo_elem / RoRBM_elem);
        }
    }
    return result;
}

CRSMatrix* GetUbRandomMatrices(int N, int NumberOfBases, bool check) {
    const int B = NumberOfBases;
    const int N_N = N * N;

    CRSMatrix* UbMatrices = new CRSMatrix[B];
    MKL_Complex16** Matrices = new MKL_Complex16 * [B];
    for (int b = 0; b < B; b++) {
        Matrices[b] = new MKL_Complex16[N_N];
        GetUnitaryMatrix(42 + b * 3, N, Matrices[b]);
    }

    if (check) {
        for (int b = 0; b < B; b++) {
            TransitionMatrix::PrintMatrix(Matrices[b], N, N, "A");
            MKL_Complex16* Result = new MKL_Complex16[N_N];
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    Result[j + i * N] = MKL_Complex16(0.0, 0.0);
                    for (int k = 0; k < N; k++) {
                        Result[j + i * N] += Matrices[b][k + i * N] * Matrices[b][k + j * N];
                    }
                }
            }
            TransitionMatrix::PrintMatrix(Result, N, N, "A * A^T");
            delete[] Result;
        }
    }

    for (int b = 0; b < B; b++) {
        UbMatrices[b] = CRSMatrix(N, Matrices[b]);
    }

    for (int b = 0; b < B; b++) {
        delete[] Matrices[b];
    }
    delete[]Matrices;

    return UbMatrices;
}
