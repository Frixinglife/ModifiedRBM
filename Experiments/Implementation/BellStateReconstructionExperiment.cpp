#include "BellStateReconstructionWithMixing.h"

void BellStateReconstructionExperiment(double alpha, int N_h, int N_a, int epochs, acc_number lr, int freq) {
	int N_v = 4;
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555);
	MKL_Complex16* OriginalRoMatrix = GetBellStateDensityMatrixWithMixing(alpha);
	BellStateReconstructionWithMixing(RBM, OriginalRoMatrix, alpha, epochs, lr, freq);
}
