#include "RandomMatricesForRBM.h"

RandomMatricesForRBM::RandomMatricesForRBM(int seed) {
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
}

RandomMatricesForRBM::~RandomMatricesForRBM() {
    vslDeleteStream(&stream);
}

void RandomMatricesForRBM::GetRandomMatrix(acc_number* Matrix, int M, int N, acc_number left, acc_number right) {
    const int size = M * N;
    //left = (acc_number)-0.005, right = (acc_number)0.005;
    TRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, size, Matrix, left, right);
}

void RandomMatricesForRBM::GetRandomVector(acc_number* Vec, int N, acc_number left, acc_number right) {
    //for (int i = 0; i < N; i++) { Vec[i] = (acc_number)0.0; }
    TRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, Vec, left, right);
}
