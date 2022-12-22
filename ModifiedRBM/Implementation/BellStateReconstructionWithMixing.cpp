#include "BellStateReconstructionWithMixing.h"
#include "MatrixAndVectorOperations.h"
#include "TransitionMatrix.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>

MKL_Complex16* GetBellStateDensityMatrixWithMixing(double alpha) {
    MKL_Complex16 BellState[4];
    MKL_Complex16 Identity[16];
    MKL_Complex16 BellRo[16];

    BellState[0] = MKL_Complex16(1.0 / std::sqrt(2.0), 0.0);
    BellState[1] = MKL_Complex16(0.0, 0.0);
    BellState[2] = MKL_Complex16(0.0, 0.0);
    BellState[3] = MKL_Complex16(1.0 / std::sqrt(2.0), 0.0);

    int N = 4;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Identity[j + i * N] = MKL_Complex16(0.0, 0.0);
            BellRo[j + i * N] = BellState[i] * BellState[j];
        }
        Identity[i + i * N] = MKL_Complex16(0.25, 0.0);
    }

    MKL_Complex16* RoMatrix = new MKL_Complex16[16];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            RoMatrix[j + i * N] = (1.0 - alpha) * BellRo[j + i * N] + alpha * Identity[j + i * N];
        }
    }

    return RoMatrix;
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

void KroneckerProduct(MKL_Complex16* Matrix_A, int size_A, MKL_Complex16* Matrix_B, int size_B, MKL_Complex16* Matrix_Res) {
    int index = 0;
    for (int i = 0; i < size_A; i++) {
        for (int k = 0; k < size_B; k++) {
            for (int j = 0; j < size_A; j++) {
                for (int l = 0; l < size_B; l++) {
                    Matrix_Res[index++] = Matrix_A[j + i * size_A] * Matrix_B[l + k * size_B];
                }
            }
        }
    }
}

CRSMatrix* GetUbMatrices() {
    MKL_Complex16 sigma_x[4] = {
        MKL_Complex16(0.0, 0.0),
        MKL_Complex16(1.0, 0.0),
        MKL_Complex16(1.0, 0.0),
        MKL_Complex16(0.0, 0.0)
    };

    MKL_Complex16 sigma_y[4] = {
        MKL_Complex16(0.0, 0.0),
        MKL_Complex16(0.0, -1.0),
        MKL_Complex16(0.0, 1.0),
        MKL_Complex16(0.0, 0.0)
    };

    MKL_Complex16 sigma_z[4] = {
        MKL_Complex16(1.0, 0.0),
        MKL_Complex16(0.0, 0.0),
        MKL_Complex16(0.0, 0.0),
        MKL_Complex16(-1.0, 0.0)
    };

    const int B = 9;
    const int N = 4;
    const int size = 2;
    const int N_N = N * N;

    CRSMatrix* UbMatrices = new CRSMatrix[B];
    MKL_Complex16** Matrices = new MKL_Complex16*[B];
    for (int b = 0; b < B; b++) {
        Matrices[b] = new MKL_Complex16[N_N];
    }

    KroneckerProduct(sigma_x, size, sigma_x, size, Matrices[0]);
    KroneckerProduct(sigma_x, size, sigma_y, size, Matrices[1]);
    KroneckerProduct(sigma_x, size, sigma_z, size, Matrices[2]);

    KroneckerProduct(sigma_y, size, sigma_x, size, Matrices[3]);
    KroneckerProduct(sigma_y, size, sigma_y, size, Matrices[4]);
    KroneckerProduct(sigma_y, size, sigma_z, size, Matrices[5]);

    KroneckerProduct(sigma_z, size, sigma_x, size, Matrices[6]);
    KroneckerProduct(sigma_z, size, sigma_y, size, Matrices[7]);
    KroneckerProduct(sigma_z, size, sigma_z, size, Matrices[8]);

    for (int b = 0; b < B; b++) {
        UbMatrices[b] = CRSMatrix(N, Matrices[b]);
    }

    for (int b = 0; b < B; b++) {
        delete[] Matrices[b];
    }
    delete[]Matrices;

    return UbMatrices;
}

void BellStateReconstructionWithMixing(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, double alpha, int epochs, acc_number lr, int freq) {
    std::cout << "Starting the training process\n\n";
    std::cout << "Iterations:\n";

    int N = RBM.FirstModifiedRBM.N_v;
    int NumberOfBases = 9;

    CRSMatrix* UbMatrices = GetUbMatrices();
    MKL_Complex16** OriginalRoMatrices = new MKL_Complex16* [NumberOfBases];

    std::ofstream* fout_fidelity = new std::ofstream[NumberOfBases];
    std::ofstream* fout_diag_original = new std::ofstream[NumberOfBases];
    std::ofstream* fout_diag_basis = new std::ofstream[NumberOfBases];

    for (int b = 0; b < NumberOfBases; b++) {
        OriginalRoMatrices[b] = TransitionMatrix::GetNewRoMatrix(OriginalRoMatrix, UbMatrices[b], N);

        fout_fidelity[b] = std::ofstream("..\\Results\\fidelity_" + std::string(TYPE_OUT) + "_" 
            + std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

        fout_diag_original[b] = std::ofstream("..\\Results\\diag_original_" + std::string(TYPE_OUT) + "_" +
            std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

        fout_diag_basis[b] = std::ofstream("..\\Results\\diag_basis_" + std::string(TYPE_OUT) + "_" +
            std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < NumberOfBases; b++) {
        for (int l = 1; l <= epochs; l++) {
            MKL_Complex16* RoMatrix = RBM.GetRoMatrix();

            std::cout << b + 1 << " / " << NumberOfBases << ", " << l << " / " << epochs << " \n";

            if (l % freq == 0 || l == 1) {
                MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, UbMatrices[b], N);

                fout_fidelity[b] << GetFidelity(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";

                if (l == epochs) {
                    for (int i = 0; i < N; i++) {
                        fout_diag_basis[b] << NewRoMatrix[i + i * N].real() << "\n";
                        fout_diag_original[b] << OriginalRoMatrices[b][i + i * N].real() << "\n";
                    }
                }

                delete[] NewRoMatrix;
            }

            RBM.WeightMatricesUpdate(N, OriginalRoMatrices[b], RoMatrix, &UbMatrices[b], lr);
            delete[]RoMatrix;
        }
    }

    auto diff = std::chrono::high_resolution_clock::now() - start;

    for (int b = 0; b < NumberOfBases; b++) {
        fout_diag_original[b].close();
        fout_diag_basis[b].close();
        fout_fidelity[b].close();
    }

    double work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()) / 1000.0;

    std::ofstream fout_config("..\\Results\\config.txt", std::ios_base::out | std::ios_base::trunc);
    fout_config << epochs << "\n";
    fout_config << freq << "\n";
    fout_config << work_time << "\n";
    fout_config << N << "\n";
    fout_config << NumberOfBases << "\n";
    fout_config << alpha << "\n";
    fout_config.close();

    std::ofstream fout_times("..\\Results\\times_train.txt", std::ios_base::app);
    fout_times << TYPE_OUT << " " << work_time << "\n";
    fout_times.close();

    std::cout << "\nFinishing the training process\n";
    std::cout << "\nMatrix size: " << N << " x " << N << "\n";
    std::cout << "Data type: " << TYPE_OUT << "\n";
    std::cout << "Time: " << work_time << " s\n";
    std::cout << "P_dep: " << alpha << "\n";
    std::cout << "Fidelity: " << GetFidelity(N, OriginalRoMatrix, RBM.GetRoMatrix()) << "\n";

    delete[]fout_fidelity;
    delete[]UbMatrices;
}

void BellStateReconstructionWithMixingForAllBasis(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, 
    double alpha, int epochs, acc_number lr, int freq) {

    std::cout << "Starting the training process\n\n";
    std::cout << "Iterations:\n";

    int N = RBM.FirstModifiedRBM.N_v;

    CRSMatrix* UbMatrices = GetUbMatrices();
    int NumberOfBases = 9;

    MKL_Complex16** OriginalRoMatrices = new MKL_Complex16 * [NumberOfBases];

    for (int b = 0; b < NumberOfBases; b++) {
        OriginalRoMatrices[b] = TransitionMatrix::GetNewRoMatrix(OriginalRoMatrix, UbMatrices[b], N);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int l = 1; l <= epochs; l++) {
        MKL_Complex16* RoMatrix = RBM.GetRoMatrix();

        std::cout << l << " / " << epochs << " \n";

        RBM.WeightMatricesUpdate(N, OriginalRoMatrices, RoMatrix, NumberOfBases, UbMatrices, lr);
        delete[]RoMatrix;
    }

    auto diff = std::chrono::high_resolution_clock::now() - start;

    double work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()) / 1000.0;

    std::cout << "\nFinishing the training process\n";
    std::cout << "\nMatrix size: " << N << " x " << N << "\n";
    std::cout << "Data type: " << TYPE_OUT << "\n";
    std::cout << "Time: " << work_time << " s\n";
    std::cout << "Fidelity: " << GetFidelity(N, OriginalRoMatrix, RBM.GetRoMatrix()) << "\n";

    delete[]UbMatrices;
}