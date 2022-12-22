#ifndef _BELL_STATE_RECONSTRUCTION_EXPERIMENT_H_
#define _BELL_STATE_RECONSTRUCTION_EXPERIMENT_H_

#include "DataType.h"

void BellStateReconstructionExperiment(double alpha, int N_h, int N_a, int epochs, acc_number lr, int freq);
void BellStateReconstructionExperimentForAllBasis(double alpha, int N_h, int N_a, int epochs, acc_number lr, int freq);

#endif //_BELL_STATE_RECONSTRUCTION_EXPERIMENT_H_
