#include "MatrixAndVectorOperations.h"
#include "TrainingProcedure.h"
#include "TransitionMatrix.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>

double NormMatrixDiag(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* RoMatrixRBM) {
    double result = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = OriginalRoMatrix[i + i * N].real() - RoMatrixRBM[i + i * N].real();
        result += diff * diff;
    }
    return std::sqrt(result);
}

double MaxEigDiffMarix(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* RoMatrixRBM) {
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

double KullbachLeiblerNorm(int N, MKL_Complex16** OriginalRoMatrices, MKL_Complex16* RoMatrixRBM, int NumberOfBases, CRSMatrix* UbMatrices) {
    double result = 0.0;
    for (int b = 0; b < NumberOfBases; b++) {
        MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrixRBM, UbMatrices[b], N);
        for (int i = 0; i < N; i++) {
            double OrigRo_elem = OriginalRoMatrices[b][i + i * N].real();
            double RoRBM_elem = NewRoMatrix[i + i * N].real();
            if (std::abs(OrigRo_elem) >= 1e-8) {
                result += OrigRo_elem * std::log(OrigRo_elem / RoRBM_elem);
            }
        }
    }
    return result;
}

void TrainingProcedure(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, int NumberOfBases, int NumberOfUnitary, int epochs, acc_number lr, int freq) {
    std::cout << "Starting the training process\n\n";
    std::cout << "Iterations:\n";

    int N = RBM.FirstModifiedRBM.N_v;

    CRSMatrix* UbMatrices = new CRSMatrix[NumberOfBases];

    MKL_Complex16** OriginalRoMatrices = new MKL_Complex16*[NumberOfBases];

    std::ofstream fout_diag_norm("..\\Results\\diag_norm_" + std::string(TYPE_OUT) + ".txt", std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_eig_norm("..\\Results\\eig_norm_" + std::string(TYPE_OUT) + ".txt", std::ios_base::out | std::ios_base::trunc);

    std::ofstream* fout_diag_norms = new std::ofstream[NumberOfBases];
    std::ofstream* fout_eig_norms = new std::ofstream[NumberOfBases];

    for (int b = 0; b < NumberOfBases; b++) {
        TransitionMatrix TM(30);
        UbMatrices[b] = TM.GetCRSTransitionMatrix(N, NumberOfUnitary, b);
        OriginalRoMatrices[b] = TransitionMatrix::GetNewRoMatrix(OriginalRoMatrix, UbMatrices[b], N);

        fout_diag_norms[b] = std::ofstream("..\\Results\\diag_norm_" + std::string(TYPE_OUT) + "_" + 
            std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

        fout_eig_norms[b] = std::ofstream("..\\Results\\eig_norm_" + std::string(TYPE_OUT) + "_" + 
            std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);
    }

    std::ofstream fout_kullbach_leibler_norm("..\\Results\\kullbach_leibler_norm_" + std::string(TYPE_OUT) + ".txt", std::ios_base::out | std::ios_base::trunc);

    auto start = std::chrono::high_resolution_clock::now();

    for (int l = 1; l <= epochs; l++) {
        MKL_Complex16* RoMatrix = RBM.GetRoMatrix();

        std::cout << l << " / " << epochs << " \n";

        if (l % freq == 0 || l == 1) {
            fout_kullbach_leibler_norm << KullbachLeiblerNorm(N, OriginalRoMatrices, RoMatrix, NumberOfBases, UbMatrices) << "\n";
            fout_diag_norm << NormMatrixDiag(N, OriginalRoMatrix, RoMatrix) << "\n";
            fout_eig_norm << MaxEigDiffMarix(N, OriginalRoMatrix, RoMatrix) << "\n";

            for (int b = 0; b < NumberOfBases; b++) {
                MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, UbMatrices[b], N);
                fout_diag_norms[b] << NormMatrixDiag(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
                fout_eig_norms[b] << MaxEigDiffMarix(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
                delete[] NewRoMatrix;
            }

            //std::ofstream fout("..\\Results\\Train\\matrix_diag_" + std::string(TYPE_OUT) + "_" + std::to_string(l) + ".txt",
            //    std::ios_base::out | std::ios_base::trunc);
            //for (int i = 0; i < N; i++) {
            //    fout << RoMatrix[i + i * N].real() << "\n";
            //}
            //fout.close();
        }

        RBM.WeightMatricesUpdate(N, OriginalRoMatrices, RoMatrix, NumberOfBases, UbMatrices, lr);
        delete[]RoMatrix;
    }

    auto diff = std::chrono::high_resolution_clock::now() - start;

    fout_kullbach_leibler_norm.close();
    fout_diag_norm.close();
    fout_eig_norm.close();
    for (int b = 0; b < NumberOfBases; b++) {
        fout_diag_norms[b].close();
        fout_eig_norms[b].close();
    }

    double work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()) / 1000.0;

    std::ofstream fout_config("..\\Results\\config.txt", std::ios_base::out | std::ios_base::trunc);
    fout_config << epochs << "\n";
    fout_config << freq << "\n";
    fout_config << work_time << "\n";
    fout_config << N << "\n";
    fout_config << NumberOfBases << "\n";
    fout_config.close();

    std::ofstream fout_times("..\\Results\\times_train.txt", std::ios_base::app);
    fout_times << TYPE_OUT << " " << work_time << "\n";
    fout_times.close();

    std::cout << "\nFinishing the training process\n";
    std::cout << "\nMatrix size: " << N << " x " << N << "\n";
    std::cout << "Data type: " << TYPE_OUT << "\n";
    std::cout << "Time: " << work_time << " s\n";

    delete[]fout_diag_norms;
    delete[]fout_eig_norms;
    delete[]UbMatrices;
}

double KullbachLeiblerNorm(int N, MKL_Complex16* OriginalRoMatrix, MKL_Complex16* NewRoMatrixRBM) {
    double result = 0.0;
    for (int i = 0; i < N; i++) {
        double OrigRo_elem = OriginalRoMatrix[i + i * N].real();
        double RoRBM_elem = NewRoMatrixRBM[i + i * N].real();
        if (std::abs(OrigRo_elem) >= 1e-8) {
            result += OrigRo_elem * std::log(OrigRo_elem / RoRBM_elem);
        }
    }
    return result;
}

void TrainingProcedureSeparatelyForBases(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, int NumberOfBases, 
    int NumberOfUnitary, int epochs, acc_number lr, int freq) {

    std::cout << "Starting the training process\n\n";
    std::cout << "Iterations:\n";

    int N = RBM.FirstModifiedRBM.N_v;

    CRSMatrix* UbMatrices = new CRSMatrix[NumberOfBases];
    MKL_Complex16** OriginalRoMatrices = new MKL_Complex16*[NumberOfBases];

    std::ofstream* fout_kullbach_leibler_norms = new std::ofstream[NumberOfBases];
    std::ofstream* fout_diag_norms = new std::ofstream[NumberOfBases];
    std::ofstream* fout_eig_norms = new std::ofstream[NumberOfBases];

    std::ofstream fout_diag_norm("..\\Results\\diag_norm_" + std::string(TYPE_OUT) + ".txt", 
        std::ios_base::out | std::ios_base::trunc);

    std::ofstream fout_eig_norm("..\\Results\\eig_norm_" + std::string(TYPE_OUT) + ".txt", 
        std::ios_base::out | std::ios_base::trunc);

    std::ofstream fout_kullbach_leibler_norm("..\\Results\\kullbach_leibler_norm_" 
        + std::string(TYPE_OUT) + ".txt", std::ios_base::out | std::ios_base::trunc);

    for (int b = 0; b < NumberOfBases; b++) {
        TransitionMatrix TM(30);
        UbMatrices[b] = TM.GetCRSTransitionMatrix(N, NumberOfUnitary, b);
        OriginalRoMatrices[b] = TransitionMatrix::GetNewRoMatrix(OriginalRoMatrix, UbMatrices[b], N);

        fout_kullbach_leibler_norms[b] = std::ofstream("..\\Results\\kullbach_leibler_norm_" 
            + std::string(TYPE_OUT) + "_" + std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

       fout_diag_norms[b] = std::ofstream("..\\Results\\diag_norm_" 
           + std::string(TYPE_OUT) + "_" + std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

       fout_eig_norms[b] = std::ofstream("..\\Results\\eig_norm_" 
           + std::string(TYPE_OUT) + "_" + std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < NumberOfBases; b++) {
        for (int l = 1; l <= epochs; l++) {
            MKL_Complex16* RoMatrix = RBM.GetRoMatrix();

            std::cout << b + 1 << " / " << NumberOfBases << ", " << l << " / " << epochs << " \n";

            if (l % freq == 0 || l == 1) {
                MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, UbMatrices[b], N);
                
                fout_kullbach_leibler_norms[b] << KullbachLeiblerNorm(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
                fout_diag_norms[b] << NormMatrixDiag(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
                fout_eig_norms[b] << MaxEigDiffMarix(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";

                delete[] NewRoMatrix;

                if (b == NumberOfBases - 1) {
                    fout_kullbach_leibler_norm << KullbachLeiblerNorm(N, OriginalRoMatrices, RoMatrix, NumberOfBases, UbMatrices) << "\n";
                    fout_diag_norm << NormMatrixDiag(N, OriginalRoMatrix, RoMatrix) << "\n";
                    fout_eig_norm << MaxEigDiffMarix(N, OriginalRoMatrix, RoMatrix) << "\n";
                }
            }

            RBM.WeightMatricesUpdate(N, OriginalRoMatrices[b], RoMatrix, &UbMatrices[b], lr);
            delete[]RoMatrix;
        }
    }

    auto diff = std::chrono::high_resolution_clock::now() - start;

    fout_kullbach_leibler_norm.close();
    fout_diag_norm.close();
    fout_eig_norm.close();

    for (int b = 0; b < NumberOfBases; b++) {
        fout_kullbach_leibler_norms[b].close();
        fout_diag_norms[b].close();
        fout_eig_norms[b].close();
    }

    double work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()) / 1000.0;

    std::ofstream fout_config("..\\Results\\config.txt", std::ios_base::out | std::ios_base::trunc);
    fout_config << epochs << "\n";
    fout_config << freq << "\n";
    fout_config << work_time << "\n";
    fout_config << N << "\n";
    fout_config << NumberOfBases << "\n";
    fout_config.close();

    std::ofstream fout_times("..\\Results\\times_train.txt", std::ios_base::app);
    fout_times << TYPE_OUT << " " << work_time << "\n";
    fout_times.close();

    std::cout << "\nFinishing the training process\n";
    std::cout << "\nMatrix size: " << N << " x " << N << "\n";
    std::cout << "Data type: " << TYPE_OUT << "\n";
    std::cout << "Time: " << work_time << " s\n";

    delete[]fout_kullbach_leibler_norms;
    delete[]fout_diag_norms;
    delete[]fout_eig_norms;
    delete[]UbMatrices;
}
