#include "TransitionMatrixCRSExperiments.h"
#include "TransitionMatrixExperiments.h"
#include "CRSMatrixExperiments.h"
#include "RoMatrixExperiments.h"
#include "TrainingExperiments.h"

int main() {
    int N_v, N_h, N_a;
    N_v = N_h = N_a = 32;
    int NumberOfBases = 5;
    int NumberOfUnitary = 1;
    acc_number lr = (acc_number)1e-2;
    int epochs = 2000;
    int freq = 2;

    //TrainingExperiment(N_v, N_h, N_a, NumberOfBases, NumberOfUnitary, epochs, lr, freq);
    TrainingExperimentSeparatelyForBases(N_v, N_h, N_a, NumberOfBases, NumberOfUnitary, epochs, lr, freq);

    return 0;
}
