#include "TransitionMatrixCRSExperiments.h"
#include "TransitionMatrixExperiments.h"
#include "CRSMatrixExperiments.h"
#include "RoMatrixExperiments.h"
#include "NeuralDensityOperators.h"
#include "TrainingProcedure.h"
#include <iostream>

int main() {
    int N_v, N_h, N_a;
    N_v = N_h = N_a = 32;
    //GetWorkTime(N_v, N_h, N_a, true);

    NeuralDensityOperators DensityOperators(N_v, N_h, N_a);
    acc_number lr = (acc_number)1e-2;
    int epochs = 200;
    int freq = 50;

    TrainingProcedure(DensityOperators, epochs, lr, freq);

    return 0;
}
