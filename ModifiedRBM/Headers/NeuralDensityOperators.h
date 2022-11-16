#ifndef _NEURAL_DENSITY_OPERATORS_H_
#define _NEURAL_DENSITY_OPERATORS_H_

#include "ComplexMKL.h"
#include "DataType.h"
#include "ModifiedRBM.h"
#include "CRSMatrix.h"

class NeuralDensityOperators {
public:
    ModifiedRBM FirstModifiedRBM, SecondModifiedRBM;

    NeuralDensityOperators(int N_v, int N_h, int N_a, int seed = 42);
    ~NeuralDensityOperators() {};

    void PrintRBMs() const;

    double GetGamma(int N, acc_number* FirstSigma, acc_number* SecondSigma, char PlusOrMinus);
    MKL_Complex16 GetPi(int N, acc_number* FirstSigma, acc_number* SecondSigma);
    MKL_Complex16 GetRoWithoutExp(int N, acc_number* FirstSigma, acc_number* SecondSigma);
    MKL_Complex16* GetRoMatrix(double *work_time = nullptr, bool plot = false);

    acc_number Sigmoid(acc_number arg);
    TComplex Sigmoid(TComplex arg);

    acc_number GetGammaGrad(int N, acc_number* FirstSigma, acc_number* SecondSigma, 
        int i, int j, char LambdaOrMu, char Variable);
    TComplex GetPiGrad(int N, acc_number* FirstSigma, acc_number* SecondSigma, 
        int i, int j, char LambdaOrMu, char Variable);
    acc_number GetLogRoGrad(int N, acc_number* Sigma, int i, int j, char Variable);
    
    acc_number WeightSumRo(int N, MKL_Complex16* Ro, int ind_i, int ind_j, char Variable);
    TComplex WeightSumLambdaMu(int N, MKL_Complex16* Ro, CRSMatrix& Ub, int ind_sum, 
        int ind_i, int ind_j, char LambdaOrMu, char Variable);

    TComplex GetGradLambdaMu(int N, MKL_Complex16* OriginalRo, MKL_Complex16* Ro, CRSMatrix& Ub, int ind_i, int ind_j,
        char LambdaOrMu, char Variable);

    void WeightUpdate(int N, MKL_Complex16* OriginalRo, MKL_Complex16* Ro, CRSMatrix& Ub, acc_number lr);
};

#endif //_NEURAL_DENSITY_OPERATORS_H_
