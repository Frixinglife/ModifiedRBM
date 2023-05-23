#include "BellStateReconstructionWithMixing.h"
#include "MetricsAndHelperFunctions.h"

void BellStateReconstructionExperiment(double alpha, int N_h, int N_a, int NumberOfBases, int epochs, acc_number lr, int freq) {
	int N_v = 4;
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555); //"identity"
	MKL_Complex16* OriginalRoMatrix = GetBellStateDensityMatrixWithMixing(alpha);//GetRandomDiagMatrix(N_v);
	BellStateReconstructionWithMixing(RBM, OriginalRoMatrix, alpha, NumberOfBases, epochs, lr, freq);
	delete[] OriginalRoMatrix;
}

void BellStateReconstructionExperimentForAllBasis(double alpha, int N_h, int N_a, int NumberOfBases, int epochs, acc_number lr, int freq) {
	int N_v = 4;
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555); //"identity"
	MKL_Complex16* OriginalRoMatrix = GetBellStateDensityMatrixWithMixing(alpha);//GetRandomDiagMatrix(N_v);
	BellStateReconstructionWithMixingForAllBasis(RBM, OriginalRoMatrix, alpha, NumberOfBases, epochs, lr, freq);
	delete[] OriginalRoMatrix;
}
