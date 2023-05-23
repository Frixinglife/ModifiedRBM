#include "TrainingProcedure.h"
#include "MetricsAndHelperFunctions.h"

void TrainingExperiment(int N_v, int N_h, int N_a, int NumberOfBases, int NumberOfUnitary, int epochs, acc_number lr, int freq) {
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555, "identity");

	NeuralDensityOperators SupportRBM(N_v, N_h, N_a, 777);
	MKL_Complex16* OriginalRoMatrix = SupportRBM.GetRoMatrix(nullptr, false);

	TrainingProcedure(RBM, OriginalRoMatrix, NumberOfBases, NumberOfUnitary, epochs, lr, freq);

	delete[] OriginalRoMatrix;
}

void TrainingExperimentSeparatelyForBases(int N_v, int N_h, int N_a, int NumberOfBases, int NumberOfUnitary, int epochs, acc_number lr, int freq) {
	NeuralDensityOperators RBM(N_v, N_h, N_a, 555, "identity");

	NeuralDensityOperators SupportRBM(N_v, N_h, N_a, 777);
	MKL_Complex16* OriginalRoMatrix = SupportRBM.GetRoMatrix(nullptr, false);

	TrainingProcedureSeparatelyForBases(RBM, OriginalRoMatrix, NumberOfBases, NumberOfUnitary, epochs, lr, freq);

	delete[] OriginalRoMatrix;
}
