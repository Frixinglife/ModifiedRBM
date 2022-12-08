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
    void Sigmoid(acc_number* arg, int N);
    void Sigmoid(TComplex* arg, int N);

    acc_number* GetGammaGrad(int N, acc_number* FirstSigma, acc_number* SecondSigma,
        char LambdaOrMu, char Variable);
    TComplex* GetPiGrad(int N, acc_number* FirstSigma, acc_number* SecondSigma,
        char LambdaOrMu, char Variable);
    acc_number* GetLogRoGrad(int N, acc_number* Sigma, char Variable);
    acc_number* WeightSumRo(int N, MKL_Complex16* Ro, char Variable);
    TComplex* WeightSumLambdaMu(int N, MKL_Complex16** OriginalRo, MKL_Complex16* Ro, int NumberOfBases, 
        CRSMatrix* UbMatrices, char LambdaOrMu, char Variable);
    TComplex* GetGradLambdaMu(int N, MKL_Complex16** OriginalRo, MKL_Complex16* Ro, int NumberOfBases, 
        CRSMatrix* UbMatrices, char LambdaOrMu, char Variable);

    void WeightMatricesUpdate(int N, MKL_Complex16** OriginalRo, MKL_Complex16* Ro, int NumberOfBases, CRSMatrix* UbMatrices, acc_number lr);
};

#endif //_NEURAL_DENSITY_OPERATORS_H_
