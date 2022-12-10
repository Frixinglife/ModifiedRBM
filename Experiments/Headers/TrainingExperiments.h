#ifndef _TRAINING_EXPERIMENTS_H_
#define _TRAINING_EXPERIMENTS_H_

#include "DataType.h"

void TrainingExperiment(int N_v, int N_h, int N_a, int NumberOfBases, int NumberOfUnitary, int epochs, acc_number lr, int freq);
void TrainingExperimentSeparatelyForBases(int N_v, int N_h, int N_a, int NumberOfBases, int NumberOfUnitary, int epochs, acc_number lr, int freq);

#endif //_TRAINING_EXPERIMENTS_H_
