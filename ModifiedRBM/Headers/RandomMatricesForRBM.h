#ifndef _RANDOM_MATRICES_FOR_RBM_H_
#define _RANDOM_MATRICES_FOR_RBM_H_

#include "ComplexMKL.h"
#include "DataType.h"

class RandomMatricesForRBM {
private:
    VSLStreamStatePtr stream;

public:
    RandomMatricesForRBM(int seed = 42);
    ~RandomMatricesForRBM();

    void GetRandomMatrix(acc_number* Matrix, int M, int N, acc_number left = (acc_number)0.0, 
        acc_number right = (acc_number)1.0);

    void GetRandomVector(acc_number* Vec, int N, acc_number left = (acc_number)0.0, 
        acc_number right = (acc_number)1.0);
};

#endif //_RANDOM_MATRICES_FOR_RBM_H_
