#include "TrainingProcedure.h"
#include "TransitionMatrix.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

void TrainingProcedure(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, int epochs, acc_number lr, int freq) {
    std::cout << "Starting the training process\n\n";
    std::cout << "Iterations:\n";

    int N = RBM.FirstModifiedRBM.N_v;

    TransitionMatrix TM;
    CRSMatrix Ub = TM.GetCRSTransitionMatrix(N);

    auto start = std::chrono::high_resolution_clock::now();

    for (int l = 1; l <= epochs; l++) {
        MKL_Complex16* RoMatrix = RBM.GetRoMatrix();

        std::cout << l << " / " << epochs << " \n";

        if (l % freq == 0 || l == 1) {
            //TransitionMatrix::PrintMatrix(NewRoMatrix, N, N, "New ro matrix");

            std::ofstream fout("..\\Results\\Train\\matrix_diag_" + std::string(TYPE_OUT) + "_" + std::to_string(l) + ".txt",
                std::ios_base::out | std::ios_base::trunc);

            for (int i = 0; i < N; i++) {
                fout << RoMatrix[i + i * N].real() << "\n";
            }

            fout.close();
        }

        RBM.WeightUpdate(N, OriginalRoMatrix, RoMatrix, Ub, lr);

        delete[]RoMatrix;
    }

    auto diff = std::chrono::high_resolution_clock::now() - start;

    double work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()) / 1000.0;

    std::ofstream fout_config("..\\Results\\Train\\config.txt", std::ios_base::out | std::ios_base::trunc);

    fout_config << epochs << "\n";
    fout_config << freq << "\n";
    fout_config << work_time << "\n";

    fout_config.close();

    std::cout << "\nFinishing the training process\n";
    std::cout << "\nMatrix size: " << N << " x " << N << "\n";
    std::cout << "Data type: " << TYPE_OUT << "\n";
    std::cout << "Time: " << work_time << " s\n";
}
