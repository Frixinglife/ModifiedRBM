#include "BellStateReconstructionExperiment.h"
#include "TransitionMatrixCRSExperiments.h"
#include "TransitionMatrixExperiments.h"
#include "CRSMatrixExperiments.h"
#include "RoMatrixExperiments.h"
#include "TrainingExperiments.h"

int main() {
    //int N_v, N_h, N_a;
    //N_v = 32;
    //N_h = N_a = 4;
    //int NumberOfBases = 5;
    //int NumberOfUnitary = 1;
    //acc_number lr = (acc_number)1e-2;
    //int epochs = 2000;
    //int freq = 2;

    //TrainingExperimentSeparatelyForBases(N_v, N_h, N_a, NumberOfBases, NumberOfUnitary, epochs, lr, freq);

    double alpha = 0.5;
    int N_h = 1;
    int N_a = 2;
    acc_number lr = (acc_number)1e-2;
    int epochs = 2000;
    int freq = 2;

    BellStateReconstructionExperiment(alpha, N_h, N_a, epochs, lr, freq);

    return 0;
}
