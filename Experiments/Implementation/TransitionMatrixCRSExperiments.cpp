#include "NeuralDensityOperators.h"
#include "TransitionMatrix.h"
#include "CRSMatrix.h"
#include <iostream>

void GetTransitionMatrixAndNewRoCRS(int N, bool show) {
    NeuralDensityOperators DensityOperators(N, N, N);
    DensityOperators.PrintRBM();

    MKL_Complex16* RoMatrix = DensityOperators.GetRoMatrix();
    TransitionMatrix::PrintMatrix(RoMatrix, N, N, "Ro matrix");

    TransitionMatrix TM;
    int NumberOfUnitary = 1;
    int IndexUnitary = 0;

    CRSMatrix Ub = TM.GetCRSTransitionMatrix(N, NumberOfUnitary, IndexUnitary, show);
    Ub.PrintCRS("Ub");

    MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, Ub, N);
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
}

void GetSamplesCRS(int N, int NumberOfSamples) {
    NeuralDensityOperators DensityOperators(N, N, N);
    TransitionMatrix TM;

    MKL_Complex16* RoMatrix = DensityOperators.GetRoMatrix();
    CRSMatrix Ub = TM.GetCRSTransitionMatrix(N);
    MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, Ub, N);

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
}
