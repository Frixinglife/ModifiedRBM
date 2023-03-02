#include "BellStateReconstructionWithMixing.h"

MKL_Complex16* GetRandomDiagMatrix(int N) {
	MKL_Complex16* Matrix = new MKL_Complex16[N * N];
	for (int i = 0; i < N * N; i++) {
		Matrix[i] = MKL_Complex16(0.0, 0.0);
	}

	acc_number* random_numbers = new acc_number[N];
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MT19937, 10);
	TRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, random_numbers, 0.0, 1.0);
	vslDeleteStream(&stream);

	acc_number sum = (acc_number)0.0;
	for (int i = 0; i < N; i++) {
		sum += random_numbers[i];
	}

	for (int i = 0; i < N; i++) {
		Matrix[i + i * N] = MKL_Complex16(random_numbers[i] / sum, 0.0);
	}
	delete[] random_numbers;

	return Matrix;
}

void BellStateReconstructionExperiment(double alpha, int N_h, int N_a, int epochs, acc_number lr, int freq) {
	int N_v = 4;
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555, "identity");
	MKL_Complex16* OriginalRoMatrix = GetBellStateDensityMatrixWithMixing(alpha);
	BellStateReconstructionWithMixing(RBM, OriginalRoMatrix, alpha, epochs, lr, freq);
}

void BellStateReconstructionExperimentForAllBasis(double alpha, int N_h, int N_a, int epochs, acc_number lr, int freq) {
	int N_v = 4;
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555, "identity");
	MKL_Complex16* OriginalRoMatrix = GetBellStateDensityMatrixWithMixing(alpha);
	BellStateReconstructionWithMixingForAllBasis(RBM, OriginalRoMatrix, alpha, epochs, lr, freq);
}
