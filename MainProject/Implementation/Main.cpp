#include "BellStateReconstructionExperiments.h"
#include "TransitionMatrixCRSExperiments.h"
#include "TransitionMatrixExperiments.h"
#include "CRSMatrixExperiments.h"
#include "RoMatrixExperiments.h"
#include "TrainingExperiments.h"

int main() {
    int N_v, N_h, N_a;
    N_v = 8;
    N_h = 1;
    N_a = 1;
    int NumberOfBases = 4;
    int NumberOfUnitary = 1;
    acc_number lr = (acc_number)1e-2;
    int epochs = 100;
    int freq = 2;

    TrainingExperiment(N_v, N_h, N_a, NumberOfBases, NumberOfUnitary, epochs, lr, freq);

    //double alpha = 1.0;
    //BellStateReconstructionExperimentForAllBasis(alpha, N_h, N_a, NumberOfBases, epochs, lr, freq);

    return 0;
}
