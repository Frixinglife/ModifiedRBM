#include "TransitionMatrixCRSExperiments.h"
#include "TransitionMatrixExperiments.h"
#include "CRSMatrixExperiments.h"
#include "RoMatrixExperiments.h"
#include "TrainingExperiments.h"

int main() {
    int N_v, N_h, N_a;
    N_v = N_h = N_a = 64;
    acc_number lr = (acc_number)1e-2;
    int epochs = 200;
    int freq = 50;

    TrainingExperiment(N_v, N_h, N_a, epochs, lr, freq);

    return 0;
}
