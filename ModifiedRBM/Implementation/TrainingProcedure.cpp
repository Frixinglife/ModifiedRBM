#include "MatrixAndVectorOperations.h"
#include "TrainingProcedure.h"
#include "MetricsAndHelperFunctions.h"
#include "TransitionMatrix.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>

void TrainingProcedure(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, int NumberOfBases, int NumberOfUnitary, int epochs, acc_number lr, int freq) {
    std::cout << "Starting the training process\n\n";
    std::cout << "Iterations:\n";

    int N = RBM.ModifiedRBM.N_v;

    CRSMatrix* UbMatrices = new CRSMatrix[NumberOfBases];
    MKL_Complex16** OriginalRoMatrices = new MKL_Complex16*[NumberOfBases];

    std::ofstream fout_diag_norm(std::string(TYPE_OUT) + "_diag_norm.txt",
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_eig_norm(std::string(TYPE_OUT) + "_eig_norm.txt", 
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_kullbach_leibler_norm(std::string(TYPE_OUT) + "_kullbach_leibler_norm.txt",
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_orig_ro_diag(std::string(TYPE_OUT) + "_orig_ro_diag.txt",
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_rbm_ro_diag(std::string(TYPE_OUT) + "_rbm_ro_diag.txt",
        std::ios_base::out | std::ios_base::trunc);

    //std::ofstream* fout_diag_norms = new std::ofstream[NumberOfBases];
    //std::ofstream* fout_eig_norms = new std::ofstream[NumberOfBases];

    for (int b = 0; b < NumberOfBases; b++) {
        TransitionMatrix TM(30);
        UbMatrices[b] = TM.GetCRSTransitionMatrix(N, NumberOfUnitary, b);
        OriginalRoMatrices[b] = TransitionMatrix::GetNewRoMatrix(OriginalRoMatrix, UbMatrices[b], N);

        //fout_diag_norms[b] = std::ofstream(std::string(TYPE_OUT) + "_diag_norm_" +
        //    std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

        //fout_eig_norms[b] = std::ofstream(std::string(TYPE_OUT) + "_eig_norm_" +
        //    std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int l = 1; l <= epochs; l++) {
        MKL_Complex16* RoMatrix = RBM.GetRoMatrix();

        std::cout << l << " / " << epochs << " \n";

        if (l % freq == 0 || l == 1) {
            fout_kullbach_leibler_norm << KullbackLeiblerNorm(N, OriginalRoMatrices, RoMatrix, NumberOfBases, UbMatrices) << "\n";
            fout_diag_norm << NormMatrixDiag(N, OriginalRoMatrix, RoMatrix) << "\n";
            fout_eig_norm << MaxEigDiffMatrix(N, OriginalRoMatrix, RoMatrix) << "\n";

            //for (int b = 0; b < NumberOfBases; b++) {
            //    MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, UbMatrices[b], N);
            //    fout_diag_norms[b] << NormMatrixDiag(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
            //    fout_eig_norms[b] << MaxEigDiffMarix(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
            //    delete[] NewRoMatrix;
            //}
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

    //for (int b = 0; b < NumberOfBases; b++) {
    //    fout_diag_norms[b].close();
    //    fout_eig_norms[b].close();
    //}

    double work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()) / 1000.0;
    double fidelity = GetFidelity(N, OriginalRoMatrix, RBM.GetRoMatrix());

    std::ofstream fout_config(std::string(TYPE_OUT) + "_config.txt", std::ios_base::out | std::ios_base::trunc);
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

    //delete[]fout_diag_norms;
    //delete[]fout_eig_norms;
    for (int i = 0; i < NumberOfBases; i++) {
        delete OriginalRoMatrices[i];
    }
    delete[]OriginalRoMatrices;
    delete[]UbMatrices;
}

void TrainingProcedureSeparatelyForBases(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, int NumberOfBases, 
    int NumberOfUnitary, int epochs, acc_number lr, int freq) {

    std::cout << "Starting the training process\n\n";
    std::cout << "Iterations:\n";

    int N = RBM.ModifiedRBM.N_v;

    CRSMatrix* UbMatrices = new CRSMatrix[NumberOfBases];
    MKL_Complex16** OriginalRoMatrices = new MKL_Complex16*[NumberOfBases];

    //std::ofstream* fout_kullbach_leibler_norms = new std::ofstream[NumberOfBases];
    //std::ofstream* fout_diag_norms = new std::ofstream[NumberOfBases];
    //std::ofstream* fout_eig_norms = new std::ofstream[NumberOfBases];

    std::ofstream fout_diag_norm(std::string(TYPE_OUT) + "_diag_norm.txt", 
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_eig_norm(std::string(TYPE_OUT) + "_eig_norm.txt", 
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_kullbach_leibler_norm(std::string(TYPE_OUT) + "_kullbach_leibler_norm.txt",
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_orig_ro_diag(std::string(TYPE_OUT) + "_orig_ro_diag.txt",
        std::ios_base::out | std::ios_base::trunc);
    std::ofstream fout_rbm_ro_diag(std::string(TYPE_OUT) + "_rbm_ro_diag.txt",
        std::ios_base::out | std::ios_base::trunc);

    for (int b = 0; b < NumberOfBases; b++) {
        TransitionMatrix TM(30);
        UbMatrices[b] = TM.GetCRSTransitionMatrix(N, NumberOfUnitary, b);
        OriginalRoMatrices[b] = TransitionMatrix::GetNewRoMatrix(OriginalRoMatrix, UbMatrices[b], N);

       // fout_kullbach_leibler_norms[b] = std::ofstream(std::string(TYPE_OUT) + "_kullbach_leibler_norm_" + 
       //     std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

       //fout_diag_norms[b] = std::ofstream(std::string(TYPE_OUT) + "_diag_norm_" + 
       //    std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);

       //fout_eig_norms[b] = std::ofstream(std::string(TYPE_OUT) + "_eig_norm_" + 
       //    std::to_string(b) + ".txt", std::ios_base::out | std::ios_base::trunc);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < NumberOfBases; b++) {
        for (int l = 1; l <= epochs; l++) {
            MKL_Complex16* RoMatrix = RBM.GetRoMatrix();

            std::cout << b + 1 << " / " << NumberOfBases << ", " << l << " / " << epochs << " \n";

            if (l % freq == 0 || l == 1) {
                //MKL_Complex16* NewRoMatrix = TransitionMatrix::GetNewRoMatrix(RoMatrix, UbMatrices[b], N);
                //fout_kullbach_leibler_norms[b] << KullbachLeiblerNorm(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
                //fout_diag_norms[b] << NormMatrixDiag(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
                //fout_eig_norms[b] << MaxEigDiffMarix(N, OriginalRoMatrices[b], NewRoMatrix) << "\n";
                //delete[] NewRoMatrix;

                if (b == NumberOfBases - 1) {
                    fout_kullbach_leibler_norm << KullbackLeiblerNorm(N, OriginalRoMatrices, RoMatrix, NumberOfBases, UbMatrices) << "\n";
                    fout_diag_norm << NormMatrixDiag(N, OriginalRoMatrix, RoMatrix) << "\n";
                    fout_eig_norm << MaxEigDiffMatrix(N, OriginalRoMatrix, RoMatrix) << "\n";
                }
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
    //    fout_diag_norms[b].close();
    //    fout_eig_norms[b].close();
    //}

    double work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()) / 1000.0;
    double fidelity = GetFidelity(N, OriginalRoMatrix, RBM.GetRoMatrix());

    std::ofstream fout_config(std::string(TYPE_OUT) + "_config.txt", std::ios_base::out | std::ios_base::trunc);
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

    //delete[]fout_kullbach_leibler_norms;
    //delete[]fout_diag_norms;
    //delete[]fout_eig_norms;
    for (int i = 0; i < NumberOfBases; i++) {
        delete OriginalRoMatrices[i];
    }
    delete[]OriginalRoMatrices;
    delete[]UbMatrices;
}
