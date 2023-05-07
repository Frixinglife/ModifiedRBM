#include "NeuralDensityOperators.h"
#include "TransitionMatrix.h"
#include <iostream>
#include <iomanip>

void CheckUnitaryMatrices(int NumberOfU, int NumberOfUnitary, int IndexUnitary) {
    TransitionMatrix TM;

    MKL_Complex16* MatricesU = new MKL_Complex16[4 * NumberOfU];
    TM.GetUnitaryMatrices(MatricesU, NumberOfU, NumberOfUnitary, IndexUnitary);
    TM.ShowUnitaryMatrices(MatricesU, NumberOfU);

    MKL_Complex16* MatricesUt = new MKL_Complex16[4 * NumberOfU];
    MKL_Complex16* Matrix = new MKL_Complex16[4];

    for (int i = 0; i < NumberOfU; i++) {
        Matrix = TM.GetHermitianConjugateMatrix(MatricesU + i * 4, 2);
        for (int j = 0; j < 4; j++) {
            MatricesUt[j + i * 4] = Matrix[j];
        }
    }

    std::cout << "U_t matrices:\n";
    for (int i = 0; i < 2 * NumberOfU; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << std::setw(30) << MatricesUt[j + i * 2];
        }
        std::cout << ((i % 2 == 1) ? "\n\n" : "\n");
    }
    std::cout << "\n";

    MKL_Complex16* Res = new MKL_Complex16[4 * NumberOfU];
    for (int i = 0; i < 4 * NumberOfU; i++) {
        Res[i] = MKL_Complex16(0.0, 0.0);
    }

    for (int count = 0; count < NumberOfU; count++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    Res[j + i * 2 + 4 * count] += TransitionMatrix::ComplexMult(
                        MatricesU[k + i * 2 + 4 * count], MatricesUt[j + k * 2 + 4 * count]);
                }
            }
        }
    }

    std::cout << "U * U_t matrices:\n";
    for (int i = 0; i < 2 * NumberOfU; i++) {
        for (int j = 0; j < 2; j++) {
            std::cout << std::setw(30) << Res[j + i * 2];
        }
        std::cout << ((i % 2 == 1) ? "\n\n" : "\n");
    }
    std::cout << "\n";

    delete[] MatricesU;
    delete[] MatricesUt;
    delete[] Matrix;
    delete[] Res;
}

void GetTransitionMatrixAndNewRo(int N, bool show) {
    NeuralDensityOperators DensityOperators(N, N, N);
    DensityOperators.PrintRBM();

    MKL_Complex16* RoMatrix = DensityOperators.GetRoMatrix();
    TransitionMatrix::PrintMatrix(RoMatrix, N, N, "Ro matrix");

    TransitionMatrix TM;
    int NumberOfUnitary = 1;
    int IndexUnitary = 0;

    MKL_Complex16* Ub = TM.GetTransitionMatrix(N, NumberOfUnitary, IndexUnitary, show);
    TransitionMatrix::PrintMatrix(Ub, N, N, "Ub");

    MKL_Complex16* Ub_t = TransitionMatrix::GetHermitianConjugateMatrix(Ub, N);
    TransitionMatrix::PrintMatrix(Ub_t, N, N, "Ub_t");

    MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, Ub, Ub_t, N);
    TransitionMatrix::PrintMatrix(NewRoMatrix, N, N, "New ro matrix");

    double* NewRoMatrixDiag = new double[N];
    for (int i = 0; i < N; i++) {
        NewRoMatrixDiag[i] = NewRoMatrix[i + i * N].real();
    }

    double trace = 0.0;
    for (int i = 0; i < N; i++) {
        trace += NewRoMatrixDiag[i];
    }

    std::cout << "Diag new ro matrix:\n";
    for (int i = 0; i < N; i++) {
        std::cout << NewRoMatrixDiag[i] << "\n";
    }
    std::cout << "\n";

    std::cout << "Trace: " << trace << "\n\n";

    for (int i = 0; i < N; i++) {
        NewRoMatrixDiag[i] /= trace;
    }

    std::cout << "Diag new ro matrix after normalization:\n";
    for (int i = 0; i < N; i++) {
        std::cout << NewRoMatrixDiag[i] << "\n";
    }
    std::cout << "\n";

    delete[]RoMatrix;
    delete[]NewRoMatrix;
    delete[]NewRoMatrixDiag;
    delete[]Ub;
    delete[]Ub_t;
}

void GetSamples(int N, int NumberOfSamples) {
    NeuralDensityOperators DensityOperators(N, N, N);
    TransitionMatrix TM;

    MKL_Complex16* RoMatrix = DensityOperators.GetRoMatrix();
    MKL_Complex16* Ub = TM.GetTransitionMatrix(N);
    MKL_Complex16* Ub_t = TransitionMatrix::GetHermitianConjugateMatrix(Ub, N);
    MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, Ub, Ub_t, N);

    double* NewRoMatrixDiag = new double[N];
    for (int i = 0; i < N; i++) {
        NewRoMatrixDiag[i] = NewRoMatrix[i + i * N].real();
    }

    double trace = 0.0;
    for (int i = 0; i < N; i++) {
        trace += NewRoMatrixDiag[i];
    }

    for (int i = 0; i < N; i++) {
        NewRoMatrixDiag[i] /= trace;
    }

    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 42);
    double* random_numbers = new double[NumberOfSamples];
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, NumberOfSamples, random_numbers, 0.0, 1.0);
    vslDeleteStream(&stream);
    
    int* Samples = new int[NumberOfSamples];
    for (int j = 0; j < NumberOfSamples; j++) {
        Samples[j] = 0;
        double prob_sum = 0.0;
        for (int i = 0; i < N; i++) {
            prob_sum += NewRoMatrixDiag[i];
            if (random_numbers[j] < prob_sum) {
                Samples[j] = i;
                break;
            }
        }
    }

    std::cout << "Samples:\n";
    for (int i = 0; i < NumberOfSamples; i++) {
        std::cout << Samples[i] << "\n";
    }

    delete[]Samples;
    delete[]RoMatrix;
    delete[]NewRoMatrix;
    delete[]NewRoMatrixDiag;
    delete[]Ub;
    delete[]Ub_t;
}
