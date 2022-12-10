#include "TrainingProcedure.h"

void TrainingExperiment(int N_v, int N_h, int N_a, int NumberOfBases, int NumberOfUnitary, int epochs, acc_number lr, int freq) {
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555);

	NeuralDensityOperators SupportRBM(N_v, N_h, N_a, 777);
	MKL_Complex16* OriginalRoMatrix = SupportRBM.GetRoMatrix(nullptr, true);

	TrainingProcedure(RBM, OriginalRoMatrix, NumberOfBases, NumberOfUnitary, epochs, lr, freq);
}

void TrainingExperimentSeparatelyForBases(int N_v, int N_h, int N_a, int NumberOfBases, int NumberOfUnitary, int epochs, acc_number lr, int freq) {
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555);

	NeuralDensityOperators SupportRBM(N_v, N_h, N_a, 777);
	MKL_Complex16* OriginalRoMatrix = SupportRBM.GetRoMatrix(nullptr, true);

	TrainingProcedureSeparatelyForBases(RBM, OriginalRoMatrix, NumberOfBases, NumberOfUnitary, epochs, lr, freq);
}
