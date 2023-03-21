#include "BellStateReconstructionExperiments.h"
#include "TransitionMatrixCRSExperiments.h"
#include "TransitionMatrixExperiments.h"
#include "CRSMatrixExperiments.h"
#include "RoMatrixExperiments.h"
#include "TrainingExperiments.h"

int main() {
    int N_v, N_h, N_a;
    N_v = 16;
    N_h = 1;
    N_a = 1;
    int NumberOfBases = 8;
    int NumberOfUnitary = 1;
    acc_number lr = (acc_number)1e-2;
    int epochs = 10000;
    int freq = 2;

    TrainingExperiment(N_v, N_h, N_a, NumberOfBases, NumberOfUnitary, epochs, lr, freq);
    //TrainingExperimentSeparatelyForBases(N_v, N_h, N_a, NumberOfBases, NumberOfUnitary, epochs, lr, freq);
 
    //double alpha = 0.0;
    //int N_h = 1;
    //int N_a = 1;
    //acc_number lr = (acc_number)1e-2;
    //int epochs = 10000;
    //int freq = 2;

    //BellStateReconstructionExperiment(alpha, N_h, N_a, epochs, lr, freq);
    //BellStateReconstructionExperimentForAllBasis(alpha, N_h, N_a, epochs, lr, freq);

    return 0;
}
