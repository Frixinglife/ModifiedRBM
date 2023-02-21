#include "BellStateReconstructionWithMixing.h"

void BellStateReconstructionExperiment(double alpha, int N_h, int N_a, int epochs, acc_number lr, int freq) {
	int N_v = 4;
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555, "identity");
	//MKL_Complex16* OriginalRoMatrix = GetBellStateDensityMatrixWithMixing(alpha);
	MKL_Complex16* OriginalRoMatrix = new MKL_Complex16[N_v * N_v];
	for (int i = 0; i < N_v * N_v; i++) {
		OriginalRoMatrix[i] = MKL_Complex16(0.0, 0.0);
	}

	acc_number* random_numbers = new acc_number[N_v];
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MT19937, 42);
	TRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N_v, random_numbers, 0.0, 1.0);
	vslDeleteStream(&stream);
	acc_number sum = (acc_number)0.0;
	for (int i = 0; i < N_v; i++) {
		sum += random_numbers[i];
	}
	for (int i = 0; i < N_v; i++) {
		OriginalRoMatrix[i + i * N_v] = MKL_Complex16(random_numbers[i] / sum, 0.0);
	}
	delete[] random_numbers;
	BellStateReconstructionWithMixing(RBM, OriginalRoMatrix, alpha, epochs, lr, freq);
}

void BellStateReconstructionExperimentForAllBasis(double alpha, int N_h, int N_a, int epochs, acc_number lr, int freq) {
	int N_v = 4;
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555, "identity");
	MKL_Complex16* OriginalRoMatrix = GetBellStateDensityMatrixWithMixing(alpha);
	BellStateReconstructionWithMixingForAllBasis(RBM, OriginalRoMatrix, alpha, epochs, lr, freq);
}
