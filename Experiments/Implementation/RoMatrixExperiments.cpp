#include "MatrixAndVectorOperations.h"
#include "NeuralDensityOperators.h"
#include "TransitionMatrix.h"
#include <iostream>
#include <fstream>
#include <string>

void GetRoMatrixAndEig(int N_v, int N_h, int N_a, bool plot, std::string type) {
    NeuralDensityOperators DensityOperators(N_v, N_h, N_a, 42, type);

    DensityOperators.PrintRBMs();

    MKL_Complex16* RoMatrix = DensityOperators.GetRoMatrix(nullptr, plot);
    TransitionMatrix::PrintMatrix(RoMatrix, N_v, N_v, "Ro matrix");
        
    MKL_Complex16* EigVector = new MKL_Complex16[N_v];
    MatrixAndVectorOperations::FindEigMatrix(N_v, RoMatrix, EigVector);
    
    std::cout << "Eigenvalues:\n\n";

    for (int i = 0; i < N_v; i++) {
        std::cout << EigVector[i] << "\n";
    }

    delete[]EigVector;
    delete[]RoMatrix;
}

void GetWorkTime(int N_v, int N_h, int N_a, bool plot, std::string type) {
    std::ofstream fout("..\\Results\\times.txt", std::ios_base::app);

    NeuralDensityOperators DensityOperators(N_v, N_h, N_a, 42, type);

    double work_time = 0.0;
    MKL_Complex16* RoMatrix = DensityOperators.GetRoMatrix(&work_time, plot);

    std::cout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n";
    std::cout << "Matrix size: " << N_v << "\n";
    std::cout << "Data type: " << TYPE_OUT << "\n";
    std::cout << "Time: " << work_time << " s\n\n";

    fout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n";
    fout << "Matrix size: " << N_v << "\n";
    fout << "Data type: " << TYPE_OUT << "\n";
    fout << "Time: " << work_time << " s\n\n";

    fout.close();

    delete[]RoMatrix;
}
