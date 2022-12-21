#ifndef _BELL_STATE_RECONSTRUCTION_WITH_MIXING_H_
#define _BELL_STATE_RECONSTRUCTION_WITH_MIXING_H_

#include "NeuralDensityOperators.h"

MKL_Complex16* GetBellStateDensityMatrixWithMixing(double alpha);
void BellStateReconstructionWithMixing(NeuralDensityOperators& RBM, MKL_Complex16* OriginalRoMatrix, 
	double alpha, int epochs, acc_number lr, int freq);

#endif _BELL_STATE_RECONSTRUCTION_WITH_MIXING_H_
