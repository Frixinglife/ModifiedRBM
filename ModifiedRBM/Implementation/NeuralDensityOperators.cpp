#include "MatrixAndVectorOperations.h"
#include "NeuralDensityOperators.h"
#include "RandomMatricesForRBM.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>
#include "omp.h"

#define STOCHASTIC_COEF 0.25

acc_number ONE = (acc_number)1.0;
acc_number HALF = (acc_number)0.5;
acc_number ZERO = (acc_number)0.0;

NeuralDensityOperators::NeuralDensityOperators(int N_v, int N_h, int N_a, int seed, std::string type) {
    vslNewStream(&stream_noise, VSL_BRNG_MT19937, 42 + seed);
    vslNewStream(&stream_stochastic, VSL_BRNG_MT19937, 18 + seed);

    const int N_h_N_v = N_h * N_v;
    const int N_a_N_v = N_a * N_v;

    acc_number* W_l = new acc_number[N_h_N_v];
    acc_number* W_m = new acc_number[N_h_N_v];
    acc_number* V_l = new acc_number[N_a_N_v];
    acc_number* V_m = new acc_number[N_a_N_v];
    acc_number* b_l = new acc_number[N_v];
    acc_number* b_m = new acc_number[N_v];
    acc_number* c_l = new acc_number[N_h];
    acc_number* c_m = new acc_number[N_h];
    acc_number* d = new acc_number[N_a];
    
    if (type == "identity") {
        for (int i = 0; i < N_h; i++) {
            for (int j = 0; j < N_v; j++) {
                W_l[j + i * N_v] = ZERO;
                W_m[j + i * N_v] = ZERO;
            }
        }

        for (int i = 0; i < N_a; i++) {
            for (int j = 0; j < N_v; j++) {
                V_l[j + i * N_v] = ZERO;
                V_m[j + i * N_v] = ZERO;
            }
        }

        for (int i = 0; i < N_v; i++) {
            b_l[i] = ZERO;
            b_m[i] = ZERO;
        }

        for (int i = 0; i < N_h; i++) {
            c_l[i] = ZERO;
            c_m[i] = ZERO;
        }

        for (int i = 0; i < N_a; i++) {
            d[i] = ZERO;
        }
    } else {
        RandomMatricesForRBM Random(seed);

        Random.GetRandomMatrix(W_l, N_h, N_v);
        Random.GetRandomMatrix(W_m, N_h, N_v);
        Random.GetRandomMatrix(V_l, N_a, N_v);
        Random.GetRandomMatrix(V_m, N_a, N_v);
        Random.GetRandomVector(b_l, N_v);
        Random.GetRandomVector(b_m, N_v);
        Random.GetRandomVector(c_l, N_h);
        Random.GetRandomVector(c_m, N_h);
        Random.GetRandomVector(d, N_a);
    }

    ModifiedRBM.SetModifiedRBM(N_v, N_h, N_a, W_l, W_m, V_l, V_m, b_l, b_m, c_l, c_m, d);

    delete[]W_l;
    delete[]W_m;
    delete[]V_l;
    delete[]V_m;
    delete[]b_l;
    delete[]b_m;
    delete[]c_l;
    delete[]c_m;
    delete[]d;
}

void NeuralDensityOperators::PrintRBM() const {
    ModifiedRBM.PrintModifiedRBM();
}

double NeuralDensityOperators::GetGamma(int N, acc_number* FirstSigma, acc_number* SecondSigma, char PlusOrMinus) {
    double Answer = 0.0;

    int N_h = ModifiedRBM.N_h;
    int N_v = ModifiedRBM.N_v;

    acc_number* IntermedVec = new acc_number[N];
    acc_number* FirstVec = new acc_number[N_h];
    acc_number* SecondVec = new acc_number[N_h];

    switch (PlusOrMinus) {
    case '+':
        MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, IntermedVec);
        Answer = (double)MatrixAndVectorOperations::ScalarVectorMult(N, ModifiedRBM.b_l, IntermedVec);

        MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_l , FirstSigma, FirstVec);
        MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_l, SecondSigma, SecondVec);

        MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, ModifiedRBM.c_l, FirstVec);
        MatrixAndVectorOperations::VectorsAdd(N_h, SecondVec, ModifiedRBM.c_l, SecondVec);
        
        for (int i = 0; i < N_h; i++) {
            double First = (double)FirstVec[i];
            double Second = (double)SecondVec[i];
            Answer += std::log(1.0 + std::exp(First)) + std::log(1.0 + std::exp(Second));
        }

        break;
    case '-':
        MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, IntermedVec);
        Answer = (double)MatrixAndVectorOperations::ScalarVectorMult(N, ModifiedRBM.b_m, IntermedVec);

        MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_m, FirstSigma, FirstVec);
        MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_m, SecondSigma, SecondVec);

        MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, ModifiedRBM.c_m, FirstVec);
        MatrixAndVectorOperations::VectorsAdd(N_h, SecondVec, ModifiedRBM.c_m, SecondVec);

        for (int i = 0; i < N_h; i++) {
            double First = (double)FirstVec[i];
            double Second = (double)SecondVec[i];
            Answer += std::log(1.0 + std::exp(First)) - std::log(1.0 + std::exp(Second));
        }

        break;
    default:
        break;
    }

    delete[]IntermedVec;
    delete[]FirstVec;
    delete[]SecondVec;

    Answer *= 0.5;

    return Answer;
}

MKL_Complex16 NeuralDensityOperators::GetPi(int N, acc_number* FirstSigma, acc_number* SecondSigma) {
    MKL_Complex16 Answer(0.0, 0.0);
    MKL_Complex16 One(1.0, 0.0);

    int N_a = ModifiedRBM.N_a;
    int N_v = ModifiedRBM.N_v;
    
    acc_number* Vec = new acc_number[N];
    acc_number* FirstVec = new acc_number[N_a];
    acc_number* SecondVec = new acc_number[N_a];

    MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, Vec);
    MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, ModifiedRBM.V_l, Vec, FirstVec);

    MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, Vec);
    MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, ModifiedRBM.V_m, Vec, SecondVec);

    MatrixAndVectorOperations::MultVectorByNumber(N_a, FirstVec, HALF, FirstVec);
    MatrixAndVectorOperations::MultVectorByNumber(N_a, SecondVec, HALF, SecondVec);

    MatrixAndVectorOperations::VectorsAdd(N_a, FirstVec, ModifiedRBM.d, FirstVec);

    for (int i = 0; i < N_a; i++) {
        MKL_Complex16 CurrentAnswer((double)FirstVec[i], (double)SecondVec[i]);
        Answer += std::log(One + std::exp(CurrentAnswer));
    }

    delete[]Vec;
    delete[]FirstVec;
    delete[]SecondVec;

    return Answer;
}

MKL_Complex16 NeuralDensityOperators::GetRoWithoutExp(int N, acc_number* FirstSigma, acc_number* SecondSigma) {
    MKL_Complex16 Gamma(GetGamma(N, FirstSigma, SecondSigma, '+'), GetGamma(N, FirstSigma, SecondSigma, '-'));
    MKL_Complex16 Pi = GetPi(N, FirstSigma, SecondSigma);

    return Gamma + Pi;
}

MKL_Complex16* NeuralDensityOperators::GetRoMatrix(double *work_time, bool plot) {
    int N_v = ModifiedRBM.N_v;
    const int N_v_N_v = N_v * N_v;

    MKL_Complex16* RoMatrix = new MKL_Complex16[N_v_N_v];

    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N_v; i++) {
        acc_number* FirstSigma = new acc_number[N_v];
        acc_number* SecondSigma = new acc_number[N_v];

        for (int k = 0; k < N_v; k++) {
            FirstSigma[k] = ZERO;
            SecondSigma[k] = ZERO;
        }

        for (int j = 0; j < N_v; j++) {
            FirstSigma[i] = ONE;
            SecondSigma[j] = ONE;

            RoMatrix[j + i * N_v] = GetRoWithoutExp(N_v, FirstSigma, SecondSigma);

            FirstSigma[i] = ZERO;
            SecondSigma[j] = ZERO;
        }

        delete[]FirstSigma;
        delete[]SecondSigma;
    }

    double MaxRe = 0.0;

    for (int i = 0; i < N_v; i++) {
        for (int j = 0; j < N_v; j++) {
            double CurRe = RoMatrix[j + i * N_v].real();
            if (CurRe > MaxRe) {
                MaxRe = CurRe;
            }
        }
    }

    for (int i = 0; i < N_v; i++) {
        for (int j = 0; j < N_v; j++) {
            MKL_Complex16 Elem = RoMatrix[j + i * N_v];
            MKL_Complex16 NewElem(Elem.real() - MaxRe, Elem.imag());
            RoMatrix[j + i * N_v] = std::exp(NewElem);
        }
    }

    MKL_Complex16 Sum(0.0, 0.0);

    for (int i = 0; i < N_v; i++) {
        Sum += RoMatrix[i + i * N_v];
    }

    for (int i = 0; i < N_v; i++) {
        for (int j = 0; j < N_v; j++) {
            RoMatrix[j + i * N_v] /= Sum;
        }
    }

    auto diff = std::chrono::high_resolution_clock::now() - start;

    if (work_time != nullptr) {
        *work_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()) / 1000.0;
    }   

    if (plot) {
        std::ofstream fout(MATRIX_OUT, std::ios_base::out | std::ios_base::trunc);

        for (int i = 0; i < N_v; i++) {
            fout << RoMatrix[i + i * N_v].real() << "\n";
        }

        fout.close();
    }

    return RoMatrix;
}

acc_number NeuralDensityOperators::Sigmoid(acc_number arg) {
    return ONE / (ONE + std::exp(-arg));
}

TComplex NeuralDensityOperators::Sigmoid(TComplex arg) {
    return ONE / (ONE + std::exp(-arg));
}

void NeuralDensityOperators::Sigmoid(acc_number* arg, int N) {
    for (int i = 0; i < N; i++) {
        arg[i] = Sigmoid(arg[i]);
    }
}

void NeuralDensityOperators::Sigmoid(TComplex* arg, int N) {
    for (int i = 0; i < N; i++) {
        arg[i] = Sigmoid(arg[i]);
    }
}

acc_number* NeuralDensityOperators::GetGammaGrad(int N, acc_number* FirstSigma, acc_number* SecondSigma, char LambdaOrMu, char Variable) {
    acc_number* Result = nullptr;

    int N_h = ModifiedRBM.N_h;
    int N_v = ModifiedRBM.N_v;

    acc_number* FirstVec = new acc_number[N_h];
    acc_number* SecondVec = new acc_number[N_h];

    acc_number* FirstResult;
    acc_number* SecondResult;

    if (LambdaOrMu == 'L') {
        switch (Variable) {
        case 'W':
            Result = new acc_number[N_h * N_v];

            MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_l, FirstSigma, FirstVec);
            MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_l, SecondSigma, SecondVec);

            MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, ModifiedRBM.c_l, FirstVec);
            MatrixAndVectorOperations::VectorsAdd(N_h, SecondVec, ModifiedRBM.c_l, SecondVec);

            Sigmoid(FirstVec, N_h);
            Sigmoid(SecondVec, N_h);
            
            FirstResult = new acc_number[N_h * N_v];
            SecondResult = new acc_number[N_h * N_v];
            for (int i = 0; i < N_h * N_v; i++) {
                FirstResult[i] = ZERO;
                SecondResult[i] = ZERO;
            }

            MatrixAndVectorOperations::VectorVectorMult(N_h, N_v, FirstVec, FirstSigma, FirstResult);
            MatrixAndVectorOperations::VectorVectorMult(N_h, N_v, SecondVec, SecondSigma, SecondResult);

            MatrixAndVectorOperations::MatrixAdd(N_h, N_v, FirstResult, SecondResult, Result);

            delete[] FirstResult;
            delete[] SecondResult;

            MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, Result, HALF);
            break;
        case 'V':
            break;
        case 'b':
            Result = new acc_number[N_v];
            MatrixAndVectorOperations::VectorsAdd(N_v, FirstSigma, SecondSigma, Result);
            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, Result, HALF);
            break;
        case 'c':
            Result = new acc_number[N_h];

            MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_l, FirstSigma, FirstVec);
            MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_l, SecondSigma, SecondVec);

            MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, ModifiedRBM.c_l, FirstVec);
            MatrixAndVectorOperations::VectorsAdd(N_h, SecondVec, ModifiedRBM.c_l, SecondVec);

            Sigmoid(FirstVec, N_h);
            Sigmoid(SecondVec, N_h);

            MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, SecondVec, Result);
            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, Result, HALF);
            break;
        case 'd':
            break;
        default:
            break;
        }
    }
    else if (LambdaOrMu == 'M') {
        switch (Variable) {
        case 'W':
            Result = new acc_number[N_h * N_v];

            MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_m, FirstSigma, FirstVec);
            MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_m, SecondSigma, SecondVec);

            MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, ModifiedRBM.c_m, FirstVec);
            MatrixAndVectorOperations::VectorsAdd(N_h, SecondVec, ModifiedRBM.c_m, SecondVec);

            Sigmoid(FirstVec, N_h);
            Sigmoid(SecondVec, N_h);

            FirstResult = new acc_number[N_h * N_v];
            SecondResult = new acc_number[N_h * N_v];
            for (int i = 0; i < N_h * N_v; i++) {
                FirstResult[i] = ZERO;
                SecondResult[i] = ZERO;
            }

            MatrixAndVectorOperations::VectorVectorMult(N_h, N_v, FirstVec, FirstSigma, FirstResult);
            MatrixAndVectorOperations::VectorVectorMult(N_h, N_v, SecondVec, SecondSigma, SecondResult);

            MatrixAndVectorOperations::MatrixSub(N_h, N_v, FirstResult, SecondResult, Result);

            delete[] FirstResult;
            delete[] SecondResult;

            MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, Result, HALF);
            break;
        case 'V':
            break;
        case 'b':
            Result = new acc_number[N_v];

            MatrixAndVectorOperations::VectorsSub(N_v, FirstSigma, SecondSigma, Result);
            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, Result, HALF);
            break;
        case 'c':
            Result = new acc_number[N_h];

            MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_m, FirstSigma, FirstVec);
            MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_m, SecondSigma, SecondVec);

            MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, ModifiedRBM.c_m, FirstVec);
            MatrixAndVectorOperations::VectorsAdd(N_h, SecondVec, ModifiedRBM.c_m, SecondVec);

            Sigmoid(FirstVec, N_h);
            Sigmoid(SecondVec, N_h);

            MatrixAndVectorOperations::VectorsSub(N_h, FirstVec, SecondVec, Result);
            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, Result, HALF);
            break;
        case 'd':
            break;
        default:
            break;
        }
    }

    delete[]FirstVec;
    delete[]SecondVec;

    return Result;
}

TComplex* NeuralDensityOperators::GetPiGrad(int N, acc_number* FirstSigma, acc_number* SecondSigma, char LambdaOrMu, char Variable) {
    TComplex* Result = nullptr;

    int N_v = ModifiedRBM.N_v;
    int N_a = ModifiedRBM.N_a;

    acc_number* SumVec = new acc_number[N];
    acc_number* DiffVec = new acc_number[N];
    acc_number* FirstVec = new acc_number[N_a];
    acc_number* SecondVec = new acc_number[N_a];
    TComplex* FirstArg = new TComplex[N_a];
    TComplex* SecondArg = new TComplex[N];

    if (LambdaOrMu == 'L') {
        switch (Variable) {
        case 'W':
            break;
        case 'V':
            Result = new TComplex[N_a * N_v];

            MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, SumVec);
            MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, DiffVec);

            MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, ModifiedRBM.V_l, SumVec, FirstVec);
            MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, ModifiedRBM.V_m, DiffVec, SecondVec);

            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, FirstVec, HALF);
            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, SecondVec, HALF);

            MatrixAndVectorOperations::VectorsAdd(N_a, FirstVec, ModifiedRBM.d, FirstVec);
            
            for (int i = 0; i < N_a; i++) {
                FirstArg[i] = TComplex(FirstVec[i], SecondVec[i]);
            }

            for (int i = 0; i < N; i++) {
                SecondArg[i] = TComplex(HALF * SumVec[i], ZERO);
            }

            Sigmoid(FirstArg, N_a);
            MatrixAndVectorOperations::VectorVectorMult(N_a, N_v, FirstArg, SecondArg, Result);
            break;
        case 'b':
            break;
        case 'c':
            break;
        case 'd':
            Result = new TComplex[N_a];

            MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, SumVec);
            MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, DiffVec);

            MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, ModifiedRBM.V_l, SumVec, FirstVec);
            MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, ModifiedRBM.V_m, DiffVec, SecondVec);

            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, FirstVec, HALF);
            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, SecondVec, HALF);

            MatrixAndVectorOperations::VectorsAdd(N_a, FirstVec, ModifiedRBM.d, FirstVec);

            for (int i = 0; i < N_a; i++) {
                FirstArg[i] = TComplex(FirstVec[i], SecondVec[i]);
            }

            Sigmoid(FirstArg, N_a);
            for (int i = 0; i < N_a; i++) {
                Result[i] = FirstArg[i];
            }
            break;
        default:
            break;
        }
    }
    else if (LambdaOrMu == 'M') {
        switch (Variable) {
        case 'W':
            break;
        case 'V':
            Result = new TComplex[N_a * N_v];

            MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, SumVec);
            MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, DiffVec);

            MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, ModifiedRBM.V_l, SumVec, FirstVec);
            MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, ModifiedRBM.V_m, DiffVec, SecondVec);

            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, FirstVec, HALF);
            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, SecondVec, HALF);

            MatrixAndVectorOperations::VectorsAdd(N_a, FirstVec, ModifiedRBM.d, FirstVec);

            for (int i = 0; i < N_a; i++) {
                FirstArg[i] = TComplex(FirstVec[i], SecondVec[i]);
            }

            for (int i = 0; i < N; i++) {
                SecondArg[i] = TComplex(ZERO, HALF * DiffVec[i]);
            }

            Sigmoid(FirstArg, N_a);
            MatrixAndVectorOperations::VectorVectorMult(N_a, N_v, FirstArg, SecondArg, Result);
            break;
        case 'b':
            break;
        case 'c':
            break;
        case 'd':
            break;
        default:
            break;
        }
    }

    delete[]SumVec;
    delete[]DiffVec;
    delete[]FirstVec;
    delete[]SecondVec;
    delete[]FirstArg;
    delete[]SecondArg;

    return Result;
}

acc_number* NeuralDensityOperators::GetLogRoGrad(int N, acc_number* Sigma, char Variable) {
    acc_number* Result = nullptr;
    acc_number* Vec = nullptr;

    int N_h = ModifiedRBM.N_h;
    int N_v = ModifiedRBM.N_v;
    int N_a = ModifiedRBM.N_a;

    switch (Variable) {
    case 'W':
        Result = new acc_number[N_h * N_v];
        Vec = new acc_number[N_h];

        MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_l, Sigma, Vec);
        MatrixAndVectorOperations::VectorsAdd(N_h, Vec, ModifiedRBM.c_l, Vec);
        Sigmoid(Vec, N_h);
        MatrixAndVectorOperations::VectorVectorMult(N_h, N_v, Vec, Sigma, Result);

        delete[]Vec;
        break;
    case 'V':
        Result = new acc_number[N_a * N_v];
        Vec = new acc_number[N_a];

        MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, ModifiedRBM.V_l, Sigma, Vec);
        MatrixAndVectorOperations::VectorsAdd(N_a, Vec, ModifiedRBM.d, Vec);
        Sigmoid(Vec, N_a);
        MatrixAndVectorOperations::VectorVectorMult(N_a, N_v, Vec, Sigma, Result);

        delete[]Vec;
        break;
    case 'b':
        Result = new acc_number[N_v];
        for (int i = 0; i < N_v; i++) {
            Result[i] = Sigma[i];
        }
        break;
    case 'c':
        Result = new acc_number[N_h];
        MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, ModifiedRBM.W_l, Sigma, Result);
        MatrixAndVectorOperations::VectorsAdd(N_h, Result, ModifiedRBM.c_l, Result);
        Sigmoid(Result, N_h);
        break;
    case 'd':
        Result = new acc_number[N_a];
        MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, ModifiedRBM.V_l, Sigma, Result);
        MatrixAndVectorOperations::VectorsAdd(N_a, Result, ModifiedRBM.d, Result);
        Sigmoid(Result, N_a);
        break;
    default:
        break;
    }

    return Result;
}

acc_number* NeuralDensityOperators::WeightSumRo(int N, MKL_Complex16* Ro, char Variable) {
    acc_number* Result = nullptr;

    int N_h = ModifiedRBM.N_h;
    int N_v = ModifiedRBM.N_v;
    int N_a = ModifiedRBM.N_a;

    acc_number Sum = ZERO;
    for (int i = 0; i < N; i++) {
        Sum += (acc_number)Ro[i + i * N].real();
    }
    Sum = ONE / Sum;
    

    acc_number* Sigma = new acc_number[N];
    for (int i = 0; i < N; i++) {
        Sigma[i] = ZERO;
    }

    int size = 0;
    switch (Variable) {
    case 'W':
        size = N_h * N_v;
        break;
    case 'V':
        size = N_a * N_v;
        break;
    case 'b':
        size = N_v;
        break;
    case 'c':
        size = N_h;
        break;
    case 'd':
        size = N_a;
        break;
    default:
        break;
    }

    Result = new acc_number[size];
    for (int i = 0; i < size; i++) {
        Result[i] = ZERO;
    }

    switch (Variable) {
    case 'W':
        for (int i = 0; i < N; i++) {
            Sigma[i] = ONE;
            acc_number coef = (acc_number)Ro[i + i * N].real();
            acc_number* LogRoGrad = GetLogRoGrad(N, Sigma, 'W');
            MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, LogRoGrad, coef);
            MatrixAndVectorOperations::MatrixAdd(N_h, N_v, Result, LogRoGrad, Result);
            Sigma[i] = ZERO;
            delete[]LogRoGrad;
        }
        MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, Result, Sum);
        break;
    case 'V':
        for (int i = 0; i < N; i++) {
            Sigma[i] = ONE;
            acc_number coef = (acc_number)Ro[i + i * N].real();
            acc_number* LogRoGrad = GetLogRoGrad(N, Sigma, 'V');
            MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_a, N_v, LogRoGrad, coef);
            MatrixAndVectorOperations::MatrixAdd(N_a, N_v, Result, LogRoGrad, Result);
            Sigma[i] = ZERO;
            delete[]LogRoGrad;
        }
        MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_a, N_v, Result, Sum);
        break;
    case 'b':
        for (int i = 0; i < N; i++) {
            Sigma[i] = ONE;
            acc_number coef = (acc_number)Ro[i + i * N].real();
            acc_number* LogRoGrad = GetLogRoGrad(N, Sigma, 'b');
            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, LogRoGrad, coef);
            MatrixAndVectorOperations::VectorsAdd(N_v, Result, LogRoGrad, Result);
            Sigma[i] = ZERO;
            delete[]LogRoGrad;
        }
        MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, Result, Sum);
        break;
    case 'c':
        for (int i = 0; i < N; i++) {
            Sigma[i] = ONE;
            acc_number coef = (acc_number)Ro[i + i * N].real();
            acc_number* LogRoGrad = GetLogRoGrad(N, Sigma, 'c');
            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, LogRoGrad, coef);
            MatrixAndVectorOperations::VectorsAdd(N_h, Result, LogRoGrad, Result);
            Sigma[i] = ZERO;
            delete[]LogRoGrad;
        }
        MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, Result, Sum);
        break;
    case 'd':
        for (int i = 0; i < N; i++) {
            Sigma[i] = ONE;
            acc_number coef = (acc_number)Ro[i + i * N].real();
            acc_number* LogRoGrad = GetLogRoGrad(N, Sigma, 'd');
            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, LogRoGrad, coef);
            MatrixAndVectorOperations::VectorsAdd(N_a, Result, LogRoGrad, Result);
            Sigma[i] = ZERO;
            delete[]LogRoGrad;
        }
        MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, Result, Sum);
        break;
    default:
        break;
    }

    delete[] Sigma;

    return Result;
}

TComplex* NeuralDensityOperators::WeightSumLambdaMu(int N, MKL_Complex16** OriginalRo, MKL_Complex16* Ro, int NumberOfBases,
    CRSMatrix* UbMatrices, char LambdaOrMu, char Variable) {

    int N_h = ModifiedRBM.N_h;
    int N_v = ModifiedRBM.N_v;
    int N_a = ModifiedRBM.N_a;

    int size = 0;
    switch (Variable) {
    case 'W':
        size = N_h * N_v;
        break;
    case 'V':
        size = N_a * N_v;
        break;
    case 'b':
        size = N_v;
        break;
    case 'c':
        size = N_h;
        break;
    case 'd':
        size = N_a;
        break;
    default:
        break;
    }

    TComplex* Result = new TComplex[size];
    for (int i = 0; i < size; i++) {
        Result[i] = TComplex(ZERO, ZERO);
    }

    int Random_size = (int)(STOCHASTIC_COEF * N);
    int* Random_choice = new int[Random_size];
    int* All_choice = new int[N];

    for (int b = 0; b < NumberOfBases; b++) {
        for (int i = 0; i < N; i++) {
            All_choice[i] = i;
        }
        int Curr_size = N, Random_ind = 0;
        for (int i = 0; i < Random_size; i++) {
            viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_stochastic, 1, &Random_ind, 0, Curr_size);
            Random_choice[i] = All_choice[Random_ind];
            std::swap(All_choice[Random_ind], All_choice[Curr_size - 1]);
            Curr_size--;
        }

        for (int i_rand = 0; i_rand < Random_size; i_rand++) {
            int i_n = Random_choice[i_rand];
            TComplex Sum(ZERO, ZERO);
            for (int i = UbMatrices[b].rowPtr[i_n]; i < UbMatrices[b].rowPtr[i_n + 1]; i++) {
                for (int j = UbMatrices[b].rowPtr[i_n]; j < UbMatrices[b].rowPtr[i_n + 1]; j++) {
                    int col_i = UbMatrices[b].colIndex[i];
                    int col_j = UbMatrices[b].colIndex[j];
                    TComplex CSR_val_i((acc_number)UbMatrices[b].val[i].real(), (acc_number)UbMatrices[b].val[i].imag());
                    TComplex CSR_val_j((acc_number)UbMatrices[b].val[j].real(), (acc_number)-UbMatrices[b].val[j].imag());
                    Sum += CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                }
            }
            acc_number eps = (acc_number)1e-10;
            if (Sum.real() < eps && Sum.imag() < eps) {
                continue;
            }
            Sum = TComplex((acc_number)OriginalRo[b][i_n + i_n * N].real(), ZERO) / Sum;

            acc_number* FirstSigma = new acc_number[N];
            acc_number* SecondSigma = new acc_number[N];
            for (int i = 0; i < N; i++) {
                FirstSigma[i] = ZERO;
                SecondSigma[i] = ZERO;
            }

            TComplex* LocalResult = new TComplex[size];
            for (int i = 0; i < size; i++) {
                LocalResult[i] = TComplex(ZERO, ZERO);
            }

            if (LambdaOrMu == 'L') {
                switch (Variable) {
                case 'W':
                    for (int i = UbMatrices[b].rowPtr[i_n]; i < UbMatrices[b].rowPtr[i_n + 1]; i++) {
                        for (int j = UbMatrices[b].rowPtr[i_n]; j < UbMatrices[b].rowPtr[i_n + 1]; j++) {
                            int col_i = UbMatrices[b].colIndex[i];
                            int col_j = UbMatrices[b].colIndex[j];
                            TComplex CSR_val_i((acc_number)UbMatrices[b].val[i].real(), (acc_number)UbMatrices[b].val[i].imag());
                            TComplex CSR_val_j((acc_number)UbMatrices[b].val[j].real(), (acc_number)-UbMatrices[b].val[j].imag());
                            FirstSigma[col_i] = ONE;
                            SecondSigma[col_j] = ONE;

                            TComplex* LocalGrad = new TComplex[size];
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] = TComplex(ZERO, ZERO);
                            }
                            TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'L', 'W');
                            if (PiGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += PiGrad[ind];
                                }
                            }
                            acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'L', 'W');
                            if (GammaGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += TComplex(GammaGrad[ind], ZERO);
                                }
                            }

                            TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                            MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, LocalGrad, coef);
                            MatrixAndVectorOperations::MatrixAdd(N_h, N_v, LocalResult, LocalGrad, LocalResult);

                            FirstSigma[col_i] = ZERO;
                            SecondSigma[col_j] = ZERO;

                            delete[] GammaGrad;
                            delete[] PiGrad;
                            delete[] LocalGrad;
                        }
                    }
                    MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, LocalResult, Sum);
                    MatrixAndVectorOperations::MatrixSub(N_h, N_v, Result, LocalResult, Result);
                    break;
                case 'V':
                    for (int i = UbMatrices[b].rowPtr[i_n]; i < UbMatrices[b].rowPtr[i_n + 1]; i++) {
                        for (int j = UbMatrices[b].rowPtr[i_n]; j < UbMatrices[b].rowPtr[i_n + 1]; j++) {
                            int col_i = UbMatrices[b].colIndex[i];
                            int col_j = UbMatrices[b].colIndex[j];
                            TComplex CSR_val_i((acc_number)UbMatrices[b].val[i].real(), (acc_number)UbMatrices[b].val[i].imag());
                            TComplex CSR_val_j((acc_number)UbMatrices[b].val[j].real(), (acc_number)-UbMatrices[b].val[j].imag());
                            FirstSigma[col_i] = ONE;
                            SecondSigma[col_j] = ONE;

                            TComplex* LocalGrad = new TComplex[size];
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] = TComplex(ZERO, ZERO);
                            }
                            TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'L', 'V');
                            if (PiGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += PiGrad[ind];
                                }
                            }
                            acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'L', 'V');
                            if (GammaGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += TComplex(GammaGrad[ind], ZERO);
                                }
                            }

                            TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                            MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_a, N_v, LocalGrad, coef);
                            MatrixAndVectorOperations::MatrixAdd(N_a, N_v, LocalResult, LocalGrad, LocalResult);

                            FirstSigma[col_i] = ZERO;
                            SecondSigma[col_j] = ZERO;

                            delete[] GammaGrad;
                            delete[] PiGrad;
                            delete[] LocalGrad;
                        }
                    }
                    MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_a, N_v, LocalResult, Sum);
                    MatrixAndVectorOperations::MatrixSub(N_a, N_v, Result, LocalResult, Result);
                    break;
                case 'b':
                    for (int i = UbMatrices[b].rowPtr[i_n]; i < UbMatrices[b].rowPtr[i_n + 1]; i++) {
                        for (int j = UbMatrices[b].rowPtr[i_n]; j < UbMatrices[b].rowPtr[i_n + 1]; j++) {
                            int col_i = UbMatrices[b].colIndex[i];
                            int col_j = UbMatrices[b].colIndex[j];
                            TComplex CSR_val_i((acc_number)UbMatrices[b].val[i].real(), (acc_number)UbMatrices[b].val[i].imag());
                            TComplex CSR_val_j((acc_number)UbMatrices[b].val[j].real(), (acc_number)-UbMatrices[b].val[j].imag());
                            FirstSigma[col_i] = ONE;
                            SecondSigma[col_j] = ONE;

                            TComplex* LocalGrad = new TComplex[size];
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] = TComplex(ZERO, ZERO);
                            }
                            TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'L', 'b');
                            if (PiGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += PiGrad[ind];
                                }
                            }
                            acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'L', 'b');
                            if (GammaGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += TComplex(GammaGrad[ind], ZERO);
                                }
                            }

                            TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, LocalGrad, coef);
                            MatrixAndVectorOperations::VectorsAdd(N_v, LocalResult, LocalGrad, LocalResult);

                            FirstSigma[col_i] = ZERO;
                            SecondSigma[col_j] = ZERO;

                            delete[] GammaGrad;
                            delete[] PiGrad;
                            delete[] LocalGrad;
                        }
                    }
                    MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, LocalResult, Sum);
                    MatrixAndVectorOperations::VectorsSub(N_v, Result, LocalResult, Result);
                    break;
                case 'c':
                    for (int i = UbMatrices[b].rowPtr[i_n]; i < UbMatrices[b].rowPtr[i_n + 1]; i++) {
                        for (int j = UbMatrices[b].rowPtr[i_n]; j < UbMatrices[b].rowPtr[i_n + 1]; j++) {
                            int col_i = UbMatrices[b].colIndex[i];
                            int col_j = UbMatrices[b].colIndex[j];
                            TComplex CSR_val_i((acc_number)UbMatrices[b].val[i].real(), (acc_number)UbMatrices[b].val[i].imag());
                            TComplex CSR_val_j((acc_number)UbMatrices[b].val[j].real(), (acc_number)-UbMatrices[b].val[j].imag());
                            FirstSigma[col_i] = ONE;
                            SecondSigma[col_j] = ONE;

                            TComplex* LocalGrad = new TComplex[size];
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] = TComplex(ZERO, ZERO);
                            }
                            TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'L', 'c');
                            if (PiGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += PiGrad[ind];
                                }
                            }
                            acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'L', 'c');
                            if (GammaGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += TComplex(GammaGrad[ind], ZERO);
                                }
                            }

                            TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, LocalGrad, coef);
                            MatrixAndVectorOperations::VectorsAdd(N_h, LocalResult, LocalGrad, LocalResult);

                            FirstSigma[col_i] = ZERO;
                            SecondSigma[col_j] = ZERO;

                            delete[] GammaGrad;
                            delete[] PiGrad;
                            delete[] LocalGrad;
                        }
                    }
                    MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, LocalResult, Sum);
                    MatrixAndVectorOperations::VectorsSub(N_h, Result, LocalResult, Result);
                    break;
                case 'd':
                    for (int i = UbMatrices[b].rowPtr[i_n]; i < UbMatrices[b].rowPtr[i_n + 1]; i++) {
                        for (int j = UbMatrices[b].rowPtr[i_n]; j < UbMatrices[b].rowPtr[i_n + 1]; j++) {
                            int col_i = UbMatrices[b].colIndex[i];
                            int col_j = UbMatrices[b].colIndex[j];
                            TComplex CSR_val_i((acc_number)UbMatrices[b].val[i].real(), (acc_number)UbMatrices[b].val[i].imag());
                            TComplex CSR_val_j((acc_number)UbMatrices[b].val[j].real(), (acc_number)-UbMatrices[b].val[j].imag());
                            FirstSigma[col_i] = ONE;
                            SecondSigma[col_j] = ONE;

                            TComplex* LocalGrad = new TComplex[size];
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] = TComplex(ZERO, ZERO);
                            }
                            TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'L', 'd');
                            if (PiGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += PiGrad[ind];
                                }
                            }
                            acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'L', 'd');
                            if (GammaGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += TComplex(GammaGrad[ind], ZERO);
                                }
                            }

                            TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, LocalGrad, coef);
                            MatrixAndVectorOperations::VectorsAdd(N_a, LocalResult, LocalGrad, LocalResult);

                            FirstSigma[col_i] = ZERO;
                            SecondSigma[col_j] = ZERO;

                            delete[] GammaGrad;
                            delete[] PiGrad;
                            delete[] LocalGrad;
                        }
                    }
                    MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, LocalResult, Sum);
                    MatrixAndVectorOperations::VectorsSub(N_a, Result, LocalResult, Result);
                    break;
                default:
                    break;
                }
            }
            else if (LambdaOrMu == 'M') {
                switch (Variable) {
                case 'W':
                    for (int i = UbMatrices[b].rowPtr[i_n]; i < UbMatrices[b].rowPtr[i_n + 1]; i++) {
                        for (int j = UbMatrices[b].rowPtr[i_n]; j < UbMatrices[b].rowPtr[i_n + 1]; j++) {
                            int col_i = UbMatrices[b].colIndex[i];
                            int col_j = UbMatrices[b].colIndex[j];
                            TComplex CSR_val_i((acc_number)UbMatrices[b].val[i].real(), (acc_number)UbMatrices[b].val[i].imag());
                            TComplex CSR_val_j((acc_number)UbMatrices[b].val[j].real(), (acc_number)-UbMatrices[b].val[j].imag());
                            FirstSigma[col_i] = ONE;
                            SecondSigma[col_j] = ONE;

                            TComplex* LocalGrad = new TComplex[size];
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] = TComplex(ZERO, ZERO);
                            }
                            TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'M', 'W');
                            if (PiGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += PiGrad[ind];
                                }
                            }
                            acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'M', 'W');
                            if (GammaGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += TComplex(ZERO, GammaGrad[ind]);
                                }
                            }

                            TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                            MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, LocalGrad, coef);
                            MatrixAndVectorOperations::MatrixAdd(N_h, N_v, LocalResult, LocalGrad, LocalResult);

                            FirstSigma[col_i] = ZERO;
                            SecondSigma[col_j] = ZERO;

                            delete[] GammaGrad;
                            delete[] PiGrad;
                            delete[] LocalGrad;
                        }
                    }
                    MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, LocalResult, Sum);
                    MatrixAndVectorOperations::MatrixSub(N_h, N_v, Result, LocalResult, Result);
                    break;
                case 'V':
                    for (int i = UbMatrices[b].rowPtr[i_n]; i < UbMatrices[b].rowPtr[i_n + 1]; i++) {
                        for (int j = UbMatrices[b].rowPtr[i_n]; j < UbMatrices[b].rowPtr[i_n + 1]; j++) {
                            int col_i = UbMatrices[b].colIndex[i];
                            int col_j = UbMatrices[b].colIndex[j];
                            TComplex CSR_val_i((acc_number)UbMatrices[b].val[i].real(), (acc_number)UbMatrices[b].val[i].imag());
                            TComplex CSR_val_j((acc_number)UbMatrices[b].val[j].real(), (acc_number)-UbMatrices[b].val[j].imag());
                            FirstSigma[col_i] = ONE;
                            SecondSigma[col_j] = ONE;

                            TComplex* LocalGrad = new TComplex[size];
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] = TComplex(ZERO, ZERO);
                            }
                            TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'M', 'V');
                            if (PiGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += PiGrad[ind];
                                }
                            }
                            acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'M', 'V');
                            if (GammaGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += TComplex(ZERO, GammaGrad[ind]);
                                }
                            }

                            TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                            MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_a, N_v, LocalGrad, coef);
                            MatrixAndVectorOperations::MatrixAdd(N_a, N_v, LocalResult, LocalGrad, LocalResult);

                            FirstSigma[col_i] = ZERO;
                            SecondSigma[col_j] = ZERO;

                            delete[] GammaGrad;
                            delete[] PiGrad;
                            delete[] LocalGrad;
                        }
                    }
                    MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_a, N_v, LocalResult, Sum);
                    MatrixAndVectorOperations::MatrixSub(N_a, N_v, Result, LocalResult, Result);
                    break;
                case 'b':
                    for (int i = UbMatrices[b].rowPtr[i_n]; i < UbMatrices[b].rowPtr[i_n + 1]; i++) {
                        for (int j = UbMatrices[b].rowPtr[i_n]; j < UbMatrices[b].rowPtr[i_n + 1]; j++) {
                            int col_i = UbMatrices[b].colIndex[i];
                            int col_j = UbMatrices[b].colIndex[j];
                            TComplex CSR_val_i((acc_number)UbMatrices[b].val[i].real(), (acc_number)UbMatrices[b].val[i].imag());
                            TComplex CSR_val_j((acc_number)UbMatrices[b].val[j].real(), (acc_number)-UbMatrices[b].val[j].imag());
                            FirstSigma[col_i] = ONE;
                            SecondSigma[col_j] = ONE;

                            TComplex* LocalGrad = new TComplex[size];
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] = TComplex(ZERO, ZERO);
                            }
                            TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'M', 'b');
                            if (PiGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += PiGrad[ind];
                                }
                            }
                            acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'M', 'b');
                            if (GammaGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += TComplex(ZERO, GammaGrad[ind]);
                                }
                            }

                            TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, LocalGrad, coef);
                            MatrixAndVectorOperations::VectorsAdd(N_v, LocalResult, LocalGrad, LocalResult);

                            FirstSigma[col_i] = ZERO;
                            SecondSigma[col_j] = ZERO;

                            delete[] GammaGrad;
                            delete[] PiGrad;
                            delete[] LocalGrad;
                        }
                    }
                    MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, LocalResult, Sum);
                    MatrixAndVectorOperations::VectorsSub(N_v, Result, LocalResult, Result);
                    break;
                case 'c':
                    for (int i = UbMatrices[b].rowPtr[i_n]; i < UbMatrices[b].rowPtr[i_n + 1]; i++) {
                        for (int j = UbMatrices[b].rowPtr[i_n]; j < UbMatrices[b].rowPtr[i_n + 1]; j++) {
                            int col_i = UbMatrices[b].colIndex[i];
                            int col_j = UbMatrices[b].colIndex[j];
                            TComplex CSR_val_i((acc_number)UbMatrices[b].val[i].real(), (acc_number)UbMatrices[b].val[i].imag());
                            TComplex CSR_val_j((acc_number)UbMatrices[b].val[j].real(), (acc_number)-UbMatrices[b].val[j].imag());
                            FirstSigma[col_i] = ONE;
                            SecondSigma[col_j] = ONE;

                            TComplex* LocalGrad = new TComplex[size];
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] = TComplex(ZERO, ZERO);
                            }
                            TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'M', 'c');
                            if (PiGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += PiGrad[ind];
                                }
                            }
                            acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'M', 'c');
                            if (GammaGrad != nullptr) {
                                for (int ind = 0; ind < size; ind++) {
                                    LocalGrad[ind] += TComplex(ZERO, GammaGrad[ind]);
                                }
                            }

                            TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                            MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, LocalGrad, coef);
                            MatrixAndVectorOperations::VectorsAdd(N_h, LocalResult, LocalGrad, LocalResult);

                            FirstSigma[col_i] = ZERO;
                            SecondSigma[col_j] = ZERO;

                            delete[] GammaGrad;
                            delete[] PiGrad;
                            delete[] LocalGrad;
                        }
                    }
                    MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, LocalResult, Sum);
                    MatrixAndVectorOperations::VectorsSub(N_h, Result, LocalResult, Result);
                    break;
                default:
                    break;
                }
            }

            delete[] FirstSigma;
            delete[] SecondSigma;
            delete[] LocalResult;
        }
    }

    delete[] Random_choice;
    delete[] All_choice;

    return Result;
}

TComplex* NeuralDensityOperators::GetGradLambdaMu(int N, MKL_Complex16** OriginalRo, MKL_Complex16* Ro, int NumberOfBases,
    CRSMatrix* UbMatrices, char LambdaOrMu, char Variable) {

    TComplex* Result = WeightSumLambdaMu(N, OriginalRo, Ro, NumberOfBases, UbMatrices, LambdaOrMu, Variable);

    if (LambdaOrMu == 'L') {
        int N_h = ModifiedRBM.N_h;
        int N_v = ModifiedRBM.N_v;
        int N_a = ModifiedRBM.N_a;

        int size = 0;
        switch (Variable) {
        case 'W':
            size = N_h * N_v;
            break;
        case 'V':
            size = N_a * N_v;
            break;
        case 'b':
            size = N_v;
            break;
        case 'c':
            size = N_h;
            break;
        case 'd':
            size = N_a;
            break;
        default:
            break;
        }

        acc_number* LocalResult = WeightSumRo(N, Ro, Variable);
        acc_number * Noise = new acc_number[size];

        TRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream_noise, size, Noise, 0.0, 1e-3);

        for (int ind = 0; ind < size; ind++) {
            Result[ind] += TComplex(LocalResult[ind], ZERO);
            Result[ind] += TComplex(Noise[ind], ZERO);
        }

        delete[] LocalResult;
        delete[] Noise;
    }

    return Result;
}

void NeuralDensityOperators::WeightMatricesUpdate(int N, MKL_Complex16** OriginalRo, MKL_Complex16* Ro, int NumberOfBases, CRSMatrix* UbMatrices, acc_number lr) {
    int N_v = ModifiedRBM.N_v;
    int N_h = ModifiedRBM.N_h;
    int N_a = ModifiedRBM.N_a;

    TComplex* W_l = GetGradLambdaMu(N, OriginalRo, Ro, NumberOfBases, UbMatrices, 'L', 'W');
    TComplex* W_m = GetGradLambdaMu(N, OriginalRo, Ro, NumberOfBases, UbMatrices, 'M', 'W');
    TComplex* V_l = GetGradLambdaMu(N, OriginalRo, Ro, NumberOfBases, UbMatrices, 'L', 'V');
    TComplex* V_m = GetGradLambdaMu(N, OriginalRo, Ro, NumberOfBases, UbMatrices, 'M', 'V');
    TComplex* b_l = GetGradLambdaMu(N, OriginalRo, Ro, NumberOfBases, UbMatrices, 'L', 'b');
    TComplex* b_m = GetGradLambdaMu(N, OriginalRo, Ro, NumberOfBases, UbMatrices, 'M', 'b');
    TComplex* c_l = GetGradLambdaMu(N, OriginalRo, Ro, NumberOfBases, UbMatrices, 'L', 'c');
    TComplex* c_m = GetGradLambdaMu(N, OriginalRo, Ro, NumberOfBases, UbMatrices, 'M', 'c');
    TComplex* d = GetGradLambdaMu(N, OriginalRo, Ro, NumberOfBases, UbMatrices, 'L', 'd');

    for (int i = 0; i < N_h; i++) {
        for (int j = 0; j < N_v; j++) {
            ModifiedRBM.W_l[j + i * N_v] -= lr * W_l[j + i * N_v].real();
            ModifiedRBM.W_m[j + i * N_v] -= lr * W_m[j + i * N_v].real();
        }
    }
    
    for (int i = 0; i < N_a; i++) {
        for (int j = 0; j < N_v; j++) {
            ModifiedRBM.V_l[j + i * N_v] -= lr * V_l[j + i * N_v].real();
            ModifiedRBM.V_m[j + i * N_v] -= lr * V_m[j + i * N_v].real();
        }
    }

    for (int i = 0; i < N_v; i++) {
        ModifiedRBM.b_l[i] -= lr * b_l[i].real();
        ModifiedRBM.b_m[i] -= lr * b_m[i].real();
    }

    for (int i = 0; i < N_h; i++) {
        ModifiedRBM.c_l[i] -= lr * c_l[i].real();
        ModifiedRBM.c_m[i] -= lr * c_m[i].real();
    }

    for (int i = 0; i < N_a; i++) {
        ModifiedRBM.d[i] -= lr * d[i].real();
    }

    delete[]W_l;
    delete[]W_m;
    delete[]V_l;
    delete[]V_m;
    delete[]b_l;
    delete[]b_m;
    delete[]c_l;
    delete[]c_m;
    delete[]d;
}

TComplex* NeuralDensityOperators::WeightSumLambdaMu(int N, MKL_Complex16* OriginalRo, MKL_Complex16* Ro, CRSMatrix* UbMatrix, 
    char LambdaOrMu, char Variable) {

    int N_h = ModifiedRBM.N_h;
    int N_v = ModifiedRBM.N_v;
    int N_a = ModifiedRBM.N_a;

    int size = 0;
    switch (Variable) {
    case 'W':
        size = N_h * N_v;
        break;
    case 'V':
        size = N_a * N_v;
        break;
    case 'b':
        size = N_v;
        break;
    case 'c':
        size = N_h;
        break;
    case 'd':
        size = N_a;
        break;
    default:
        break;
    }

    TComplex* Result = new TComplex[size];
    for (int i = 0; i < size; i++) {
        Result[i] = TComplex(ZERO, ZERO);
    }

    int* All_choice = new int[N];
    for (int i = 0; i < N; i++) {
        All_choice[i] = i;
    }

    int Random_size = (int)(STOCHASTIC_COEF * N);
    int* Random_choice = new int [Random_size];
    int Curr_size = N, Random_ind = 0;
    for (int i = 0; i < Random_size; i++) {
        viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_stochastic, 1, &Random_ind, 0, Curr_size);
        Random_choice[i] = All_choice[Random_ind];
        std::swap(All_choice[Random_ind], All_choice[Curr_size - 1]);
        Curr_size--;
    }

    //for (int i = 0; i < Random_size; i++) {
    //    std::cout << Random_choice[i] << " ";
    //}
    //std::cout << "\n";

    for (int i_rand = 0; i_rand < Random_size; i_rand++) {
        int i_n = Random_choice[i_rand];
        TComplex Sum(ZERO, ZERO);
        for (int i = UbMatrix->rowPtr[i_n]; i < UbMatrix->rowPtr[i_n + 1]; i++) {
            for (int j = UbMatrix->rowPtr[i_n]; j < UbMatrix->rowPtr[i_n + 1]; j++) {
                int col_i = UbMatrix->colIndex[i];
                int col_j = UbMatrix->colIndex[j];
                TComplex CSR_val_i((acc_number)UbMatrix->val[i].real(), (acc_number)UbMatrix->val[i].imag());
                TComplex CSR_val_j((acc_number)UbMatrix->val[j].real(), (acc_number)-UbMatrix->val[j].imag());
                Sum += CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
            }
        }
        acc_number eps = (acc_number)1e-10;
        if (Sum.real() < eps && Sum.imag() < eps) {
            continue;
        }
        Sum = TComplex((acc_number)OriginalRo[i_n + i_n * N].real(), ZERO) / Sum;

        acc_number* FirstSigma = new acc_number[N];
        acc_number* SecondSigma = new acc_number[N];
        for (int i = 0; i < N; i++) {
            FirstSigma[i] = ZERO;
            SecondSigma[i] = ZERO;
        }

        TComplex* LocalResult = new TComplex[size];
        for (int i = 0; i < size; i++) {
            LocalResult[i] = TComplex(ZERO, ZERO);
        }

        if (LambdaOrMu == 'L') {
            switch (Variable) {
            case 'W':
                for (int i = UbMatrix->rowPtr[i_n]; i < UbMatrix->rowPtr[i_n + 1]; i++) {
                    for (int j = UbMatrix->rowPtr[i_n]; j < UbMatrix->rowPtr[i_n + 1]; j++) {
                        int col_i = UbMatrix->colIndex[i];
                        int col_j = UbMatrix->colIndex[j];
                        TComplex CSR_val_i((acc_number)UbMatrix->val[i].real(), (acc_number)UbMatrix->val[i].imag());
                        TComplex CSR_val_j((acc_number)UbMatrix->val[j].real(), (acc_number)-UbMatrix->val[j].imag());
                        FirstSigma[col_i] = ONE;
                        SecondSigma[col_j] = ONE;

                        TComplex* LocalGrad = new TComplex[size];
                        for (int ind = 0; ind < size; ind++) {
                            LocalGrad[ind] = TComplex(ZERO, ZERO);
                        }
                        TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'L', 'W');
                        if (PiGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += PiGrad[ind];
                            }
                        }
                        acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'L', 'W');
                        if (GammaGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += TComplex(GammaGrad[ind], ZERO);
                            }
                        }

                        TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                        MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, LocalGrad, coef);
                        MatrixAndVectorOperations::MatrixAdd(N_h, N_v, LocalResult, LocalGrad, LocalResult);

                        FirstSigma[col_i] = ZERO;
                        SecondSigma[col_j] = ZERO;

                        delete[] GammaGrad;
                        delete[] PiGrad;
                        delete[] LocalGrad;
                    }
                }
                MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, LocalResult, Sum);
                MatrixAndVectorOperations::MatrixSub(N_h, N_v, Result, LocalResult, Result);
                break;
            case 'V':
                for (int i = UbMatrix->rowPtr[i_n]; i < UbMatrix->rowPtr[i_n + 1]; i++) {
                    for (int j = UbMatrix->rowPtr[i_n]; j < UbMatrix->rowPtr[i_n + 1]; j++) {
                        int col_i = UbMatrix->colIndex[i];
                        int col_j = UbMatrix->colIndex[j];
                        TComplex CSR_val_i((acc_number)UbMatrix->val[i].real(), (acc_number)UbMatrix->val[i].imag());
                        TComplex CSR_val_j((acc_number)UbMatrix->val[j].real(), (acc_number)-UbMatrix->val[j].imag());
                        FirstSigma[col_i] = ONE;
                        SecondSigma[col_j] = ONE;

                        TComplex* LocalGrad = new TComplex[size];
                        for (int ind = 0; ind < size; ind++) {
                            LocalGrad[ind] = TComplex(ZERO, ZERO);
                        }
                        TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'L', 'V');
                        if (PiGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += PiGrad[ind];
                            }
                        }
                        acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'L', 'V');
                        if (GammaGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += TComplex(GammaGrad[ind], ZERO);
                            }
                        }

                        TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                        MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_a, N_v, LocalGrad, coef);
                        MatrixAndVectorOperations::MatrixAdd(N_a, N_v, LocalResult, LocalGrad, LocalResult);

                        FirstSigma[col_i] = ZERO;
                        SecondSigma[col_j] = ZERO;

                        delete[] GammaGrad;
                        delete[] PiGrad;
                        delete[] LocalGrad;
                    }
                }
                MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_a, N_v, LocalResult, Sum);
                MatrixAndVectorOperations::MatrixSub(N_a, N_v, Result, LocalResult, Result);
                break;
            case 'b':
                for (int i = UbMatrix->rowPtr[i_n]; i < UbMatrix->rowPtr[i_n + 1]; i++) {
                    for (int j = UbMatrix->rowPtr[i_n]; j < UbMatrix->rowPtr[i_n + 1]; j++) {
                        int col_i = UbMatrix->colIndex[i];
                        int col_j = UbMatrix->colIndex[j];
                        TComplex CSR_val_i((acc_number)UbMatrix->val[i].real(), (acc_number)UbMatrix->val[i].imag());
                        TComplex CSR_val_j((acc_number)UbMatrix->val[j].real(), (acc_number)-UbMatrix->val[j].imag());
                        FirstSigma[col_i] = ONE;
                        SecondSigma[col_j] = ONE;

                        TComplex* LocalGrad = new TComplex[size];
                        for (int ind = 0; ind < size; ind++) {
                            LocalGrad[ind] = TComplex(ZERO, ZERO);
                        }
                        TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'L', 'b');
                        if (PiGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += PiGrad[ind];
                            }
                        }
                        acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'L', 'b');
                        if (GammaGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += TComplex(GammaGrad[ind], ZERO);
                            }
                        }

                        TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                        MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, LocalGrad, coef);
                        MatrixAndVectorOperations::VectorsAdd(N_v, LocalResult, LocalGrad, LocalResult);

                        FirstSigma[col_i] = ZERO;
                        SecondSigma[col_j] = ZERO;

                        delete[] GammaGrad;
                        delete[] PiGrad;
                        delete[] LocalGrad;
                    }
                }
                MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, LocalResult, Sum);
                MatrixAndVectorOperations::VectorsSub(N_v, Result, LocalResult, Result);
                break;
            case 'c':
                for (int i = UbMatrix->rowPtr[i_n]; i < UbMatrix->rowPtr[i_n + 1]; i++) {
                    for (int j = UbMatrix->rowPtr[i_n]; j < UbMatrix->rowPtr[i_n + 1]; j++) {
                        int col_i = UbMatrix->colIndex[i];
                        int col_j = UbMatrix->colIndex[j];
                        TComplex CSR_val_i((acc_number)UbMatrix->val[i].real(), (acc_number)UbMatrix->val[i].imag());
                        TComplex CSR_val_j((acc_number)UbMatrix->val[j].real(), (acc_number)-UbMatrix->val[j].imag());
                        FirstSigma[col_i] = ONE;
                        SecondSigma[col_j] = ONE;

                        TComplex* LocalGrad = new TComplex[size];
                        for (int ind = 0; ind < size; ind++) {
                            LocalGrad[ind] = TComplex(ZERO, ZERO);
                        }
                        TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'L', 'c');
                        if (PiGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += PiGrad[ind];
                            }
                        }
                        acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'L', 'c');
                        if (GammaGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += TComplex(GammaGrad[ind], ZERO);
                            }
                        }

                        TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                        MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, LocalGrad, coef);
                        MatrixAndVectorOperations::VectorsAdd(N_h, LocalResult, LocalGrad, LocalResult);

                        FirstSigma[col_i] = ZERO;
                        SecondSigma[col_j] = ZERO;

                        delete[] GammaGrad;
                        delete[] PiGrad;
                        delete[] LocalGrad;
                    }
                }
                MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, LocalResult, Sum);
                MatrixAndVectorOperations::VectorsSub(N_h, Result, LocalResult, Result);
                break;
            case 'd':
                for (int i = UbMatrix->rowPtr[i_n]; i < UbMatrix->rowPtr[i_n + 1]; i++) {
                    for (int j = UbMatrix->rowPtr[i_n]; j < UbMatrix->rowPtr[i_n + 1]; j++) {
                        int col_i = UbMatrix->colIndex[i];
                        int col_j = UbMatrix->colIndex[j];
                        TComplex CSR_val_i((acc_number)UbMatrix->val[i].real(), (acc_number)UbMatrix->val[i].imag());
                        TComplex CSR_val_j((acc_number)UbMatrix->val[j].real(), (acc_number)-UbMatrix->val[j].imag());
                        FirstSigma[col_i] = ONE;
                        SecondSigma[col_j] = ONE;

                        TComplex* LocalGrad = new TComplex[size];
                        for (int ind = 0; ind < size; ind++) {
                            LocalGrad[ind] = TComplex(ZERO, ZERO);
                        }
                        TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'L', 'd');
                        if (PiGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += PiGrad[ind];
                            }
                        }
                        acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'L', 'd');
                        if (GammaGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += TComplex(GammaGrad[ind], ZERO);
                            }
                        }

                        TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                        MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, LocalGrad, coef);
                        MatrixAndVectorOperations::VectorsAdd(N_a, LocalResult, LocalGrad, LocalResult);

                        FirstSigma[col_i] = ZERO;
                        SecondSigma[col_j] = ZERO;

                        delete[] GammaGrad;
                        delete[] PiGrad;
                        delete[] LocalGrad;
                    }
                }
                MatrixAndVectorOperations::MultVectorByNumberInPlace(N_a, LocalResult, Sum);
                MatrixAndVectorOperations::VectorsSub(N_a, Result, LocalResult, Result);
                break;
            default:
                break;
            }
        }
        else if (LambdaOrMu == 'M') {
            switch (Variable) {
            case 'W':
                for (int i = UbMatrix->rowPtr[i_n]; i < UbMatrix->rowPtr[i_n + 1]; i++) {
                    for (int j = UbMatrix->rowPtr[i_n]; j < UbMatrix->rowPtr[i_n + 1]; j++) {
                        int col_i = UbMatrix->colIndex[i];
                        int col_j = UbMatrix->colIndex[j];
                        TComplex CSR_val_i((acc_number)UbMatrix->val[i].real(), (acc_number)UbMatrix->val[i].imag());
                        TComplex CSR_val_j((acc_number)UbMatrix->val[j].real(), (acc_number)-UbMatrix->val[j].imag());
                        FirstSigma[col_i] = ONE;
                        SecondSigma[col_j] = ONE;

                        TComplex* LocalGrad = new TComplex[size];
                        for (int ind = 0; ind < size; ind++) {
                            LocalGrad[ind] = TComplex(ZERO, ZERO);
                        }
                        TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'M', 'W');
                        if (PiGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += PiGrad[ind];
                            }
                        }
                        acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'M', 'W');
                        if (GammaGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += TComplex(ZERO, GammaGrad[ind]);
                            }
                        }

                        TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                        MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, LocalGrad, coef);
                        MatrixAndVectorOperations::MatrixAdd(N_h, N_v, LocalResult, LocalGrad, LocalResult);

                        FirstSigma[col_i] = ZERO;
                        SecondSigma[col_j] = ZERO;

                        delete[] GammaGrad;
                        delete[] PiGrad;
                        delete[] LocalGrad;
                    }
                }
                MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_h, N_v, LocalResult, Sum);
                MatrixAndVectorOperations::MatrixSub(N_h, N_v, Result, LocalResult, Result);
                break;
            case 'V':
                for (int i = UbMatrix->rowPtr[i_n]; i < UbMatrix->rowPtr[i_n + 1]; i++) {
                    for (int j = UbMatrix->rowPtr[i_n]; j < UbMatrix->rowPtr[i_n + 1]; j++) {
                        int col_i = UbMatrix->colIndex[i];
                        int col_j = UbMatrix->colIndex[j];
                        TComplex CSR_val_i((acc_number)UbMatrix->val[i].real(), (acc_number)UbMatrix->val[i].imag());
                        TComplex CSR_val_j((acc_number)UbMatrix->val[j].real(), (acc_number)-UbMatrix->val[j].imag());
                        FirstSigma[col_i] = ONE;
                        SecondSigma[col_j] = ONE;

                        TComplex* LocalGrad = new TComplex[size];
                        for (int ind = 0; ind < size; ind++) {
                            LocalGrad[ind] = TComplex(ZERO, ZERO);
                        }
                        TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'M', 'V');
                        if (PiGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += PiGrad[ind];
                            }
                        }
                        acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'M', 'V');
                        if (GammaGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += TComplex(ZERO, GammaGrad[ind]);
                            }
                        }

                        TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                        MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_a, N_v, LocalGrad, coef);
                        MatrixAndVectorOperations::MatrixAdd(N_a, N_v, LocalResult, LocalGrad, LocalResult);

                        FirstSigma[col_i] = ZERO;
                        SecondSigma[col_j] = ZERO;

                        delete[] GammaGrad;
                        delete[] PiGrad;
                        delete[] LocalGrad;
                    }
                }
                MatrixAndVectorOperations::MultMatrixByNumberInPlace(N_a, N_v, LocalResult, Sum);
                MatrixAndVectorOperations::MatrixSub(N_a, N_v, Result, LocalResult, Result);
                break;
            case 'b':
                for (int i = UbMatrix->rowPtr[i_n]; i < UbMatrix->rowPtr[i_n + 1]; i++) {
                    for (int j = UbMatrix->rowPtr[i_n]; j < UbMatrix->rowPtr[i_n + 1]; j++) {
                        int col_i = UbMatrix->colIndex[i];
                        int col_j = UbMatrix->colIndex[j];
                        TComplex CSR_val_i((acc_number)UbMatrix->val[i].real(), (acc_number)UbMatrix->val[i].imag());
                        TComplex CSR_val_j((acc_number)UbMatrix->val[j].real(), (acc_number)-UbMatrix->val[j].imag());
                        FirstSigma[col_i] = ONE;
                        SecondSigma[col_j] = ONE;

                        TComplex* LocalGrad = new TComplex[size];
                        for (int ind = 0; ind < size; ind++) {
                            LocalGrad[ind] = TComplex(ZERO, ZERO);
                        }
                        TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'M', 'b');
                        if (PiGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += PiGrad[ind];
                            }
                        }
                        acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'M', 'b');
                        if (GammaGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += TComplex(ZERO, GammaGrad[ind]);
                            }
                        }

                        TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                        MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, LocalGrad, coef);
                        MatrixAndVectorOperations::VectorsAdd(N_v, LocalResult, LocalGrad, LocalResult);

                        FirstSigma[col_i] = ZERO;
                        SecondSigma[col_j] = ZERO;

                        delete[] GammaGrad;
                        delete[] PiGrad;
                        delete[] LocalGrad;
                    }
                }
                MatrixAndVectorOperations::MultVectorByNumberInPlace(N_v, LocalResult, Sum);
                MatrixAndVectorOperations::VectorsSub(N_v, Result, LocalResult, Result);
                break;
            case 'c':
                for (int i = UbMatrix->rowPtr[i_n]; i < UbMatrix->rowPtr[i_n + 1]; i++) {
                    for (int j = UbMatrix->rowPtr[i_n]; j < UbMatrix->rowPtr[i_n + 1]; j++) {
                        int col_i = UbMatrix->colIndex[i];
                        int col_j = UbMatrix->colIndex[j];
                        TComplex CSR_val_i((acc_number)UbMatrix->val[i].real(), (acc_number)UbMatrix->val[i].imag());
                        TComplex CSR_val_j((acc_number)UbMatrix->val[j].real(), (acc_number)-UbMatrix->val[j].imag());
                        FirstSigma[col_i] = ONE;
                        SecondSigma[col_j] = ONE;

                        TComplex* LocalGrad = new TComplex[size];
                        for (int ind = 0; ind < size; ind++) {
                            LocalGrad[ind] = TComplex(ZERO, ZERO);
                        }
                        TComplex* PiGrad = GetPiGrad(N, FirstSigma, SecondSigma, 'M', 'c');
                        if (PiGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += PiGrad[ind];
                            }
                        }
                        acc_number* GammaGrad = GetGammaGrad(N, FirstSigma, SecondSigma, 'M', 'c');
                        if (GammaGrad != nullptr) {
                            for (int ind = 0; ind < size; ind++) {
                                LocalGrad[ind] += TComplex(ZERO, GammaGrad[ind]);
                            }
                        }

                        TComplex coef = CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
                        MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, LocalGrad, coef);
                        MatrixAndVectorOperations::VectorsAdd(N_h, LocalResult, LocalGrad, LocalResult);

                        FirstSigma[col_i] = ZERO;
                        SecondSigma[col_j] = ZERO;

                        delete[] GammaGrad;
                        delete[] PiGrad;
                        delete[] LocalGrad;
                    }
                }
                MatrixAndVectorOperations::MultVectorByNumberInPlace(N_h, LocalResult, Sum);
                MatrixAndVectorOperations::VectorsSub(N_h, Result, LocalResult, Result);
                break;
            default:
                break;
            }
        }

        delete[] FirstSigma;
        delete[] SecondSigma;
        delete[] LocalResult;
    }

    delete[] Random_choice;
    delete[] All_choice;

    return Result;
}

TComplex* NeuralDensityOperators::GetGradLambdaMu(int N, MKL_Complex16* OriginalRo, MKL_Complex16* Ro, CRSMatrix* UbMatrix, 
    char LambdaOrMu, char Variable) {

    TComplex* Result = WeightSumLambdaMu(N, OriginalRo, Ro, UbMatrix, LambdaOrMu, Variable);

    if (LambdaOrMu == 'L') {
        int N_h = ModifiedRBM.N_h;
        int N_v = ModifiedRBM.N_v;
        int N_a = ModifiedRBM.N_a;

        int size = 0;
        switch (Variable) {
        case 'W':
            size = N_h * N_v;
            break;
        case 'V':
            size = N_a * N_v;
            break;
        case 'b':
            size = N_v;
            break;
        case 'c':
            size = N_h;
            break;
        case 'd':
            size = N_a;
            break;
        default:
            break;
        }

        acc_number* LocalResult = WeightSumRo(N, Ro, Variable);
        acc_number* Noise = new acc_number[size];
        
        TRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream_noise, size, Noise, 0.0, 1e-3);

        for (int ind = 0; ind < size; ind++) {
            Result[ind] += TComplex(LocalResult[ind], ZERO);
            Result[ind] += TComplex(Noise[ind], ZERO);
        }

        delete[] LocalResult;
        delete[] Noise;
    }

    return Result;
}

void NeuralDensityOperators::WeightMatricesUpdate(int N, MKL_Complex16* OriginalRo, MKL_Complex16* Ro, CRSMatrix* UbMatrix, acc_number lr) {
    int N_v = ModifiedRBM.N_v;
    int N_h = ModifiedRBM.N_h;
    int N_a = ModifiedRBM.N_a;

    TComplex* W_l = GetGradLambdaMu(N, OriginalRo, Ro, UbMatrix, 'L', 'W');
    TComplex* W_m = GetGradLambdaMu(N, OriginalRo, Ro, UbMatrix, 'M', 'W');
    TComplex* V_l = GetGradLambdaMu(N, OriginalRo, Ro, UbMatrix, 'L', 'V');
    TComplex* V_m = GetGradLambdaMu(N, OriginalRo, Ro, UbMatrix, 'M', 'V');
    TComplex* b_l = GetGradLambdaMu(N, OriginalRo, Ro, UbMatrix, 'L', 'b');
    TComplex* b_m = GetGradLambdaMu(N, OriginalRo, Ro, UbMatrix, 'M', 'b');
    TComplex* c_l = GetGradLambdaMu(N, OriginalRo, Ro, UbMatrix, 'L', 'c');
    TComplex* c_m = GetGradLambdaMu(N, OriginalRo, Ro, UbMatrix, 'M', 'c');
    TComplex* d = GetGradLambdaMu(N, OriginalRo, Ro, UbMatrix, 'L', 'd');

    for (int i = 0; i < N_h; i++) {
        for (int j = 0; j < N_v; j++) {
            ModifiedRBM.W_l[j + i * N_v] -= lr * W_l[j + i * N_v].real();
            ModifiedRBM.W_m[j + i * N_v] -= lr * W_m[j + i * N_v].real();
        }
    }
    
    for (int i = 0; i < N_a; i++) {
        for (int j = 0; j < N_v; j++) {
            ModifiedRBM.V_l[j + i * N_v] -= lr * V_l[j + i * N_v].real();
            ModifiedRBM.V_m[j + i * N_v] -= lr * V_m[j + i * N_v].real();
        }
    }

    for (int i = 0; i < N_v; i++) {
        ModifiedRBM.b_l[i] -= lr * b_l[i].real();
        ModifiedRBM.b_m[i] -= lr * b_m[i].real();
    }

    for (int i = 0; i < N_h; i++) {
        ModifiedRBM.c_l[i] -= lr * c_l[i].real();
        ModifiedRBM.c_m[i] -= lr * c_m[i].real();
    }

    for (int i = 0; i < N_a; i++) {
        ModifiedRBM.d[i] -= lr * d[i].real();
    }

    delete[]W_l;
    delete[]W_m;
    delete[]V_l;
    delete[]V_m;
    delete[]b_l;
    delete[]b_m;
    delete[]c_l;
    delete[]c_m;
    delete[]d;
}
