#include "BellStateReconstructionWithMixing.h"
#include "MatrixAndVectorOperations.h"
#include "MetricsAndHelperFunctions.h"
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

void EigenvectorsBasis(MKL_Complex16* RoMatrixRBM, MKL_Complex16* VR, int N) {
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
    MKL_Complex16* Work = new MKL_Complex16[N_2];
    double* rwork = new double[N_2];

    for (int i = 0; i < N_N; i++) {
        J[i] = RoMatrixRBM[i];
    }

    zgeev(&jobvl, &jobvr, &N, J, &lda, W, VL, &ldvl, VR, &ldvr, Work, &lwork, rwork, &info);

    delete[]J;
    delete[]W;
    delete[]VL;
    delete[]Work;
    delete[]rwork;
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

    //EigenvectorsBasis(RoMatrixRBM, Matrices[0], N);

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

void BellStateReconstructionWithMixing(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, double alpha, int NumberOfBases, 
    int epochs, acc_number lr, int freq) {

    std::cout << "Starting the training process\n\n";
    std::cout << "Iterations:\n";

    int N = RBM.ModifiedRBM.N_v;

    CRSMatrix* UbMatrices = new CRSMatrix[NumberOfBases];
    MKL_Complex16** OriginalRoMatrices = new MKL_Complex16* [NumberOfBases];

    //std::ofstream* fout_fidelity = new std::ofstream[NumberOfBases];
    //std::ofstream* fout_diag_original = new std::ofstream[NumberOfBases];
    //std::ofstream* fout_diag_basis = new std::ofstream[NumberOfBases];
    //std::ofstream* fout_kullbach_leibler_norms = new std::ofstream[NumberOfBases];
    //std::ofstream* fout_diag_norms = new std::ofstream[NumberOfBases];
    //std::ofstream* fout_eig_norms = new std::ofstream[NumberOfBases];

    std::ofstream fout_kullbach_leibler_norm("..\\Results\\" + std::string(TYPE_OUT) + "\\kullbach_leibler_norm.txt", 
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_diag_norm("..\\Results\\" + std::string(TYPE_OUT) + "\\diag_norm.txt", 
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_eig_norm("..\\Results\\" + std::string(TYPE_OUT) + "\\eig_norm.txt", 
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_orig_ro_diag("..\\Results\\" + std::string(TYPE_OUT) + "\\orig_ro_diag.txt",
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_rbm_ro_diag("..\\Results\\" + std::string(TYPE_OUT) + "\\rbm_ro_diag.txt",
        std::ios_base::out | std::ios_base::trunc);

    int NumberOfUnitary = 1;
    for (int b = 0; b < NumberOfBases; b++) {
        TransitionMatrix TM(30);
        UbMatrices[b] = TM.GetCRSTransitionMatrix(N, NumberOfUnitary, b);
        OriginalRoMatrices[b] = TransitionMatrix::GetNewRoMatrix(OriginalRoMatrix, UbMatrices[b], N);

        //fout_fidelity[b] = std::ofstream("..\\Results\\" + std::string(TYPE_OUT) + "\\fidelity_" + 
        //    std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

        //fout_diag_original[b] = std::ofstream("..\\Results\\" + std::string(TYPE_OUT) + "\\diag_original_" + 
        //    std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

        //fout_diag_basis[b] = std::ofstream("..\\Results\\" + std::string(TYPE_OUT) + "\\diag_basis_" + 
        //    std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

        //fout_kullbach_leibler_norms[b] = std::ofstream("..\\Results\\" + std::string(TYPE_OUT) + "\\kullbach_leibler_norm_" + 
        //    std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

        //fout_diag_norms[b] = std::ofstream("..\\Results\\" + std::string(TYPE_OUT) + "\\diag_norm_" +
        //    std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

        //fout_eig_norms[b] = std::ofstream("..\\Results\\" + std::string(TYPE_OUT) + "\\eig_norm_" +
        //    std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < NumberOfBases; b++) {
        for (int l = 1; l <= epochs; l++) {
            MKL_Complex16* RoMatrix = RBM.GetRoMatrix();

            std::cout << b + 1 << " / " << NumberOfBases << ", " << l << " / " << epochs << " \n";

            if (l % freq == 0 || l == 1) {
                MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, UbMatrices[b], N);

                //fout_fidelity[b] << GetFidelity(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
                //fout_kullbach_leibler_norms[b] << KullbachLeiblerNorm(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
                //fout_diag_norms[b] << NormMatrixDiag(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
                //fout_eig_norms[b] << MaxEigDiffMarix(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";

                //if (l == epochs) {
                //    for (int i = 0; i < N; i++) {
                //        for (int j = 0; j < N; j++) {
                //            fout_diag_original[b] << OriginalRoMatrices[b][j + i * N].real() << "\t";
                //            fout_diag_basis[b] << NewRoMatrix[j + i * N].real() << "\t";
                //        }
                //        fout_diag_original[b] << "\n";
                //        fout_diag_basis[b] << "\n";
                //    }
                //}

                if (b == NumberOfBases - 1) {
                    fout_kullbach_leibler_norm << KullbachLeiblerNorm(N, OriginalRoMatrices, RoMatrix, NumberOfBases, UbMatrices) << "\n";
                    fout_diag_norm << NormMatrixDiag(N, OriginalRoMatrix, RoMatrix) << "\n";
                    fout_eig_norm << MaxEigDiffMatrix(N, OriginalRoMatrix, RoMatrix) << "\n";
                }

                delete[] NewRoMatrix;
            }

            RBM.WeightMatricesUpdate(N, OriginalRoMatrices[b], RoMatrix, &UbMatrices[b], lr);
            delete[]RoMatrix;
        }
    }

    auto diff = std::chrono::high_resolution_clock::now() - start;

    MKL_Complex16* RoMatrix = RBM.GetRoMatrix();
    //TransitionMatrix::PrintMatrix(OriginalRoMatrix, N, N, "Ro original");
    //TransitionMatrix::PrintMatrix(RoMatrix, N, N, "Ro RBM");
    for (int i = 0; i < N; i++) {
        fout_orig_ro_diag << OriginalRoMatrix[i + i * N].real() << "\n";
        fout_rbm_ro_diag << RoMatrix[i + i * N].real() << "\n";
    }
    delete[]RoMatrix;

    fout_kullbach_leibler_norm.close();
    fout_diag_norm.close();
    fout_eig_norm.close();
    fout_orig_ro_diag.close();
    fout_rbm_ro_diag.close();

    //for (int b = 0; b < NumberOfBases; b++) {
    //    fout_kullbach_leibler_norms[b].close();
    //    fout_diag_original[b].close();
    //    fout_diag_basis[b].close();
    //    fout_fidelity[b].close();
    //    fout_diag_norms[b].close();
    //    fout_eig_norms[b].close();
    //}

    double work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()) / 1000.0;
    double fidelity = GetFidelity(N, OriginalRoMatrix, RBM.GetRoMatrix());

    std::ofstream fout_config("..\\Results\\"+ std::string(TYPE_OUT) + "\\config.txt", std::ios_base::out | std::ios_base::trunc);
    fout_config << epochs << "\n";
    fout_config << freq << "\n";
    fout_config << work_time << "\n";
    fout_config << N << "\n";
    fout_config << NumberOfBases << "\n";
    fout_config << fidelity << "\n";
    fout_config.close();

    std::cout << "\nFinishing the training process\n";
    std::cout << "\nMatrix size: " << N << " x " << N << "\n";
    std::cout << "Data type: " << TYPE_OUT << "\n";
    std::cout << "Time: " << work_time << " s\n";
    std::cout << "P_dep: " << alpha << "\n";
    std::cout << "Fidelity: " << fidelity << "\n";

    //delete[]fout_fidelity;
    //delete[]fout_diag_original;
    //delete[]fout_diag_basis;
    //delete[]fout_kullbach_leibler_norms;
    //delete[]fout_diag_norms;
    //delete[]fout_eig_norms;
    for (int i = 0; i < NumberOfBases; i++) {
        delete OriginalRoMatrices[i];
    }
    delete[]OriginalRoMatrices;
    delete[]UbMatrices;
}

void BellStateReconstructionWithMixingForAllBasis(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, 
    double alpha, int NumberOfBases, int epochs, acc_number lr, int freq) {

    std::cout << "Starting the training process\n\n";
    std::cout << "Iterations:\n";

    int N = RBM.ModifiedRBM.N_v;

    CRSMatrix* UbMatrices = new CRSMatrix[NumberOfBases];

    std::ofstream fout_kullbach_leibler_norm("..\\Results\\" + std::string(TYPE_OUT) + "\\kullbach_leibler_norm.txt",
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_diag_norm("..\\Results\\" + std::string(TYPE_OUT) + "\\diag_norm.txt", 
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_eig_norm("..\\Results\\" + std::string(TYPE_OUT) + "\\eig_norm.txt", 
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_orig_ro_diag("..\\Results\\" + std::string(TYPE_OUT) + "\\orig_ro_diag.txt",
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_rbm_ro_diag("..\\Results\\" + std::string(TYPE_OUT) + "\\rbm_ro_diag.txt",
        std::ios_base::out | std::ios_base::trunc);

    MKL_Complex16** OriginalRoMatrices = new MKL_Complex16 * [NumberOfBases];

    int NumberOfUnitary = 1;
    for (int b = 0; b < NumberOfBases; b++) {
        TransitionMatrix TM(30);
        UbMatrices[b] = TM.GetCRSTransitionMatrix(N, NumberOfUnitary, b);
        OriginalRoMatrices[b] = TransitionMatrix::GetNewRoMatrix(OriginalRoMatrix, UbMatrices[b], N);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int l = 1; l <= epochs; l++) {
        MKL_Complex16* RoMatrix = RBM.GetRoMatrix();

        std::cout << l << " / " << epochs << " \n";
        if (l % freq == 0 || l == 1) {
            fout_kullbach_leibler_norm << KullbachLeiblerNorm(N, OriginalRoMatrices, RoMatrix, NumberOfBases, UbMatrices) << "\n";
            fout_diag_norm << NormMatrixDiag(N, OriginalRoMatrix, RoMatrix) << "\n";
            fout_eig_norm << MaxEigDiffMatrix(N, OriginalRoMatrix, RoMatrix) << "\n";
        }

        RBM.WeightMatricesUpdate(N, OriginalRoMatrices, RoMatrix, NumberOfBases, UbMatrices, lr);
        delete[]RoMatrix;
    }

    auto diff = std::chrono::high_resolution_clock::now() - start;

    MKL_Complex16* RoMatrix = RBM.GetRoMatrix();
    //TransitionMatrix::PrintMatrix(OriginalRoMatrix, N, N, "Ro original");
    //TransitionMatrix::PrintMatrix(RoMatrix, N, N, "Ro RBM");
    for (int i = 0; i < N; i++) {
        fout_orig_ro_diag << OriginalRoMatrix[i + i * N].real() << "\n";
        fout_rbm_ro_diag << RoMatrix[i + i * N].real() << "\n";
    }
    delete[]RoMatrix;

    fout_kullbach_leibler_norm.close();
    fout_diag_norm.close();
    fout_eig_norm.close();
    fout_orig_ro_diag.close();
    fout_rbm_ro_diag.close();

    double work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()) / 1000.0;
    double fidelity = GetFidelity(N, OriginalRoMatrix, RBM.GetRoMatrix());

    std::ofstream fout_config("..\\Results\\" + std::string(TYPE_OUT) + "\\config.txt", std::ios_base::out | std::ios_base::trunc);
    fout_config << epochs << "\n";
    fout_config << freq << "\n";
    fout_config << work_time << "\n";
    fout_config << N << "\n";
    fout_config << NumberOfBases << "\n";
    fout_config << fidelity << "\n";
    fout_config.close();

    std::cout << "\nFinishing the training process\n";
    std::cout << "\nMatrix size: " << N << " x " << N << "\n";
    std::cout << "Data type: " << TYPE_OUT << "\n";
    std::cout << "Time: " << work_time << " s\n";
    std::cout << "Fidelity: " << fidelity << "\n";

    for (int i = 0; i < NumberOfBases; i++) {
        delete OriginalRoMatrices[i];
    }
    delete[]OriginalRoMatrices;
    delete[]UbMatrices;
}