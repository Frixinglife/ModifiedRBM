#ifndef _TRAINING_PROCEDURE_H_
#define _TRAINING_PROCEDURE_H_

#include "NeuralDensityOperators.h"

void TrainingProcedure(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, int NumberOfBases, 
	int NumberOfUnitary, int epochs, acc_number lr, int freq);
void TrainingProcedureSeparatelyForBases(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, 
	int NumberOfBases, int NumberOfUnitary, int epochs, acc_number lr, int freq);

#endif //_TRAINING_PROCEDURE_H_
