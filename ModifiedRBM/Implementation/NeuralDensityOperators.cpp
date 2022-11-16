#include "MatrixAndVectorOperations.h"
#include "NeuralDensityOperators.h"
#include "RandomMatricesForRBM.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include "omp.h"

acc_number ONE = (acc_number)1.0;
acc_number HALF = (acc_number)0.5;
acc_number ZERO = (acc_number)0.0;

NeuralDensityOperators::NeuralDensityOperators(int N_v, int N_h, int N_a, int seed) {
    const int N_h_N_v = N_h * N_v;
    const int N_a_N_v = N_a * N_v;

    acc_number* W_1 = new acc_number[N_h_N_v];
    acc_number* V_1 = new acc_number[N_a_N_v];
    acc_number* b_1 = new acc_number[N_v];
    acc_number* c_1 = new acc_number[N_h];
    acc_number* d_1 = new acc_number[N_a];

    acc_number* W_2 = new acc_number[N_h_N_v];
    acc_number* V_2 = new acc_number[N_a_N_v];
    acc_number* b_2 = new acc_number[N_v];
    acc_number* c_2 = new acc_number[N_h];
    acc_number* d_2 = new acc_number[N_a];

    RandomMatricesForRBM Random(seed);

    Random.GetRandomMatrix(W_1, N_h, N_v);
    Random.GetRandomMatrix(W_2, N_h, N_v);

    Random.GetRandomMatrix(V_1, N_a, N_v);
    Random.GetRandomMatrix(V_2, N_a, N_v);

    Random.GetRandomVector(b_1, N_v);
    Random.GetRandomVector(b_2, N_v);

    Random.GetRandomVector(c_1, N_h);
    Random.GetRandomVector(c_2, N_h);

    Random.GetRandomVector(d_1, N_a);
    Random.GetRandomVector(d_2, N_a);

    FirstModifiedRBM.SetModifiedRBM(N_v, N_h, N_a, W_1, V_1, b_1, c_1, d_1);
    SecondModifiedRBM.SetModifiedRBM(N_v, N_h, N_a, W_2, V_2, b_2, c_2, d_2);

    delete[]W_1;
    delete[]W_2;
    delete[]V_1;
    delete[]V_2;
    delete[]b_1;
    delete[]b_2;
    delete[]c_1;
    delete[]c_2;
    delete[]d_1;
    delete[]d_2;
}

void NeuralDensityOperators::PrintRBMs() const {
    FirstModifiedRBM.PrintModifiedRBM("First modified RBM");
    SecondModifiedRBM.PrintModifiedRBM("Second modified RBM");
}

double NeuralDensityOperators::GetGamma(int N, acc_number* FirstSigma, acc_number* SecondSigma, char PlusOrMinus) {
    double Answer = 0.0;

    int N_h = FirstModifiedRBM.N_h;
    int N_v = FirstModifiedRBM.N_v;

    acc_number* IntermedVec = new acc_number[N];
    acc_number* FirstVec = new acc_number[N_h];
    acc_number* SecondVec = new acc_number[N_h];

    switch (PlusOrMinus) {
    case '+':
        MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, IntermedVec);
        Answer = (double)MatrixAndVectorOperations::ScalarVectorMult(N, FirstModifiedRBM.b, IntermedVec);

        MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, FirstModifiedRBM.W, FirstSigma, FirstVec);
        MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, FirstModifiedRBM.W, SecondSigma, SecondVec);

        MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, FirstModifiedRBM.c, FirstVec);
        MatrixAndVectorOperations::VectorsAdd(N_h, SecondVec, FirstModifiedRBM.c, SecondVec);
        
        for (int i = 0; i < N_h; i++) {
            double First = (double)FirstVec[i];
            double Second = (double)SecondVec[i];
            Answer += std::log(1.0 + std::exp(First)) + std::log(1.0 + std::exp(Second));
        }

        break;
    case '-':
        MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, IntermedVec);
        Answer = (double)MatrixAndVectorOperations::ScalarVectorMult(N, SecondModifiedRBM.b, IntermedVec);

        MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, SecondModifiedRBM.W, FirstSigma, FirstVec);
        MatrixAndVectorOperations::MatrixVectorMult(N_h, N_v, SecondModifiedRBM.W, SecondSigma, SecondVec);

        MatrixAndVectorOperations::VectorsAdd(N_h, FirstVec, SecondModifiedRBM.c, FirstVec);
        MatrixAndVectorOperations::VectorsAdd(N_h, SecondVec, SecondModifiedRBM.c, SecondVec);

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

    int N_a = FirstModifiedRBM.N_a;
    int N_v = FirstModifiedRBM.N_v;
    
    acc_number* Vec = new acc_number[N];
    acc_number* FirstVec = new acc_number[N_a];
    acc_number* SecondVec = new acc_number[N_a];

    MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, Vec);
    MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, FirstModifiedRBM.V, Vec, FirstVec);

    MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, Vec);
    MatrixAndVectorOperations::MatrixVectorMult(N_a, N_v, SecondModifiedRBM.V, Vec, SecondVec);

    MatrixAndVectorOperations::MultVectorByNumber(N_a, FirstVec, HALF, FirstVec);
    MatrixAndVectorOperations::MultVectorByNumber(N_a, SecondVec, HALF, SecondVec);

    MatrixAndVectorOperations::VectorsAdd(N_a, FirstVec, FirstModifiedRBM.d, FirstVec);

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
    int N_v = FirstModifiedRBM.N_v;
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

acc_number NeuralDensityOperators::GetGammaGrad(int N, acc_number* FirstSigma, acc_number* SecondSigma, 
    int i, int j, char LambdaOrMu, char Variable) {
    acc_number Result = ZERO;

    if (LambdaOrMu == 'L') {
        acc_number* W_i = FirstModifiedRBM.W + i * FirstModifiedRBM.N_v;
        acc_number FirstArg, SecondArg;

        switch(Variable) {
        case 'W':
            FirstArg = Sigmoid(MatrixAndVectorOperations::ScalarVectorMult(N, W_i, FirstSigma) + FirstModifiedRBM.c[i]);
            SecondArg = Sigmoid(MatrixAndVectorOperations::ScalarVectorMult(N, W_i, SecondSigma) + FirstModifiedRBM.c[i]);
            Result = HALF * (FirstArg * FirstSigma[j] + SecondArg * SecondSigma[j]);
            break;
        case 'V':
            break;
        case 'b':
            Result = HALF * (FirstSigma[i] + SecondSigma[i]);
            break;
        case 'c':
            FirstArg = Sigmoid(MatrixAndVectorOperations::ScalarVectorMult(N, W_i, FirstSigma) + FirstModifiedRBM.c[i]);
            SecondArg = Sigmoid(MatrixAndVectorOperations::ScalarVectorMult(N, W_i, SecondSigma) + FirstModifiedRBM.c[i]);
            Result = HALF * (FirstArg + SecondArg);
            break;
        case 'd':
            break;
        default:
            break;
        }
    }
    else if (LambdaOrMu == 'M') {
        acc_number* W_i = SecondModifiedRBM.W + i * SecondModifiedRBM.N_v;
        acc_number FirstArg, SecondArg;

        switch (Variable) {
        case 'W':
            FirstArg = Sigmoid(MatrixAndVectorOperations::ScalarVectorMult(N, W_i, FirstSigma) + SecondModifiedRBM.c[i]);
            SecondArg = Sigmoid(MatrixAndVectorOperations::ScalarVectorMult(N, W_i, SecondSigma) + SecondModifiedRBM.c[i]);
            Result = HALF * (FirstArg * FirstSigma[j] - SecondArg * SecondSigma[j]);
            break;
        case 'V':
            break;
        case 'b':
            Result = HALF * (FirstSigma[i] - SecondSigma[i]);
            break;
        case 'c':
            FirstArg = Sigmoid(MatrixAndVectorOperations::ScalarVectorMult(N, W_i, FirstSigma) + SecondModifiedRBM.c[i]);
            SecondArg = Sigmoid(MatrixAndVectorOperations::ScalarVectorMult(N, W_i, SecondSigma) + SecondModifiedRBM.c[i]);
            Result = HALF * (FirstArg - SecondArg);
            break;
        case 'd':
            break;
        default:
            break;
        }
    }
    
    return Result;
}

TComplex NeuralDensityOperators::GetPiGrad(int N, acc_number* FirstSigma, acc_number* SecondSigma, 
    int i, int j, char LambdaOrMu, char Variable) {
    TComplex Result(ZERO, ZERO);

    if (LambdaOrMu == 'L') {
        acc_number* V_i_l = FirstModifiedRBM.V + i * FirstModifiedRBM.N_v;
        acc_number* V_i_m = SecondModifiedRBM.V + i * SecondModifiedRBM.N_v;
        acc_number* SumVec = new acc_number[N];
        acc_number* DiffVec = new acc_number[N];
        acc_number ArgRe, ArgIm;

        switch (Variable) {
        case 'W':
            break;
        case 'V':
            MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, SumVec);
            MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, DiffVec);
            ArgRe = HALF * MatrixAndVectorOperations::ScalarVectorMult(N, V_i_l, SumVec) + FirstModifiedRBM.d[i];
            ArgIm = HALF * MatrixAndVectorOperations::ScalarVectorMult(N, V_i_m, DiffVec);
            Result = Sigmoid(TComplex(ArgRe, ArgIm)) * HALF * (FirstSigma[j] + SecondSigma[j]);
            break;
        case 'b':
            break;
        case 'c':
            break;
        case 'd':
            MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, SumVec);
            MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, DiffVec);
            ArgRe = HALF * MatrixAndVectorOperations::ScalarVectorMult(N, V_i_l, SumVec) + FirstModifiedRBM.d[i];
            ArgIm = HALF * MatrixAndVectorOperations::ScalarVectorMult(N, V_i_m, DiffVec);
            Result = Sigmoid(TComplex(ArgRe, ArgIm));
            break;
        default:
            break;
        }

        delete[] SumVec;
        delete[] DiffVec;
    }
    else if (LambdaOrMu == 'M') {
        acc_number* V_i_l = FirstModifiedRBM.V + i * FirstModifiedRBM.N_v;
        acc_number* V_i_m = SecondModifiedRBM.V + i * SecondModifiedRBM.N_v;
        acc_number* SumVec = new acc_number[N];
        acc_number* DiffVec = new acc_number[N];
        acc_number ArgRe, ArgIm;

        TComplex HALF_I((acc_number)0.0, (acc_number)0.5);

        switch (Variable) {
        case 'W':
            break;
        case 'V':
            MatrixAndVectorOperations::VectorsAdd(N, FirstSigma, SecondSigma, SumVec);
            MatrixAndVectorOperations::VectorsSub(N, FirstSigma, SecondSigma, DiffVec);
            ArgRe = HALF * MatrixAndVectorOperations::ScalarVectorMult(N, V_i_l, SumVec) + FirstModifiedRBM.d[i];
            ArgIm = HALF * MatrixAndVectorOperations::ScalarVectorMult(N, V_i_m, DiffVec);
            Result = Sigmoid(TComplex(ArgRe, ArgIm)) * HALF_I * (FirstSigma[j] - SecondSigma[j]);
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

        delete[] SumVec;
        delete[] DiffVec;
    }

    return Result;
}

acc_number NeuralDensityOperators::GetLogRoGrad(int N, acc_number* Sigma, int i, int j, char Variable) {
    acc_number Result = ZERO;

    acc_number* W_i = FirstModifiedRBM.W + i * FirstModifiedRBM.N_v;
    acc_number* V_i = FirstModifiedRBM.V + i * FirstModifiedRBM.N_v;
    acc_number Arg;

    switch (Variable) {
    case 'W':
        Arg = MatrixAndVectorOperations::ScalarVectorMult(N, W_i, Sigma) + FirstModifiedRBM.c[i];
        Result = Sigmoid(Arg) * Sigma[j];
        break;
    case 'V':
        Arg = MatrixAndVectorOperations::ScalarVectorMult(N, V_i, Sigma) + FirstModifiedRBM.d[i];
        Result = Sigmoid(Arg) * Sigma[j];
        break;
    case 'b':
        Result = Sigma[i];
        break;
    case 'c':
        Arg = MatrixAndVectorOperations::ScalarVectorMult(N, W_i, Sigma) + FirstModifiedRBM.c[i];
        Result = Sigmoid(Arg);
        break;
    case 'd':
        Arg = MatrixAndVectorOperations::ScalarVectorMult(N, V_i, Sigma) + FirstModifiedRBM.d[i];
        Result = Sigmoid(Arg);
        break;
    default:
        break;
    }

    return Result;
}

acc_number NeuralDensityOperators::WeightSumRo(int N, MKL_Complex16* Ro, int ind_i, int ind_j, char Variable) {
    acc_number Sum = ZERO;
    for (int i = 0; i < N; i++) {
        Sum += (acc_number)Ro[i + i * N].real();
    }

    acc_number Result = ZERO;

    acc_number* Sigma = new acc_number[N];

    for (int i = 0; i < N; i++) {
        Sigma[i] = ZERO;
    }

    for (int i = 0; i < N; i++) {
        Sigma[i] = ONE;
        Result += (acc_number)Ro[i + i * N].real() * GetLogRoGrad(N, Sigma, ind_i, ind_j, Variable);
        Sigma[i] = ZERO;
    }

    delete[] Sigma;

    Result /= Sum;
    
    return Result;
}

TComplex NeuralDensityOperators::WeightSumLambdaMu(int N, MKL_Complex16* Ro, CRSMatrix& Ub, 
    int ind_sum, int ind_i, int ind_j, char LambdaOrMu, char Variable) {

    TComplex Sum(ZERO, ZERO);

    for (int i = Ub.rowPtr[ind_sum]; i < Ub.rowPtr[ind_sum + 1]; i++) {
        for (int j = Ub.rowPtr[ind_sum]; j < Ub.rowPtr[ind_sum + 1]; j++) {
            int col_i = Ub.colIndex[i];
            int col_j = Ub.colIndex[j];
            TComplex CSR_val_i((acc_number)Ub.val[i].real(), (acc_number)-Ub.val[i].imag());
            TComplex CSR_val_j((acc_number)Ub.val[j].real(), (acc_number)Ub.val[j].imag());
            Sum += CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N];
        }
    }

    TComplex Result(ZERO, ZERO);

    acc_number* FirstSigma = new acc_number[N];
    acc_number* SecondSigma = new acc_number[N];

    for (int i = 0; i < N; i++) {
        FirstSigma[i] = ZERO;
        SecondSigma[i] = ZERO;
    }

    for (int i = Ub.rowPtr[ind_sum]; i < Ub.rowPtr[ind_sum + 1]; i++) {
        for (int j = Ub.rowPtr[ind_sum]; j < Ub.rowPtr[ind_sum + 1]; j++) {
            int col_i = Ub.colIndex[i];
            int col_j = Ub.colIndex[j];
            TComplex CSR_val_i((acc_number)Ub.val[i].real(), (acc_number)-Ub.val[i].imag());
            TComplex CSR_val_j((acc_number)Ub.val[j].real(), (acc_number)Ub.val[j].imag());

            FirstSigma[col_i] = ONE;
            SecondSigma[col_j] = ONE;

            TComplex Arg(GetGammaGrad(N, FirstSigma, SecondSigma, ind_i, ind_j, LambdaOrMu, Variable), ZERO);
            Arg += GetPiGrad(N, FirstSigma, SecondSigma, ind_i, ind_j, LambdaOrMu, Variable);

            Result += CSR_val_i * CSR_val_j * (TComplex)Ro[col_j + col_i * N] * Arg;

            FirstSigma[col_i] = ZERO;
            SecondSigma[col_j] = ZERO;
        }
    }

    delete[] FirstSigma;
    delete[] SecondSigma;

    Result /= Sum;

    return Result;
}

TComplex NeuralDensityOperators::GetGradLambdaMu(int N, MKL_Complex16* OriginalRo, MKL_Complex16* Ro, CRSMatrix& Ub, int ind_i, int ind_j,
    char LambdaOrMu, char Variable) {

    TComplex Result(ZERO, ZERO);

    for (int i = 0; i < N; i++) {
        Result -= (TComplex)OriginalRo[i + i * N] * WeightSumLambdaMu(N, Ro, Ub, i, ind_i, ind_j, LambdaOrMu, Variable);
    }

    if (LambdaOrMu == 'L') {
        Result += WeightSumRo(N, Ro, ind_i, ind_j, Variable);
    }

    return Result;
}

void NeuralDensityOperators::WeightUpdate(int N, MKL_Complex16* OriginalRo, MKL_Complex16* Ro, CRSMatrix& Ub, acc_number lr) {
    int N_v = FirstModifiedRBM.N_v;
    int N_h = FirstModifiedRBM.N_h;
    int N_a = FirstModifiedRBM.N_a;

    //std::cout << "Grads for W (Lambda):\n";
    for (int i = 0; i < N_h; i++) {
        for (int j = 0; j < N_v; j++) {
            TComplex grad = GetGradLambdaMu(N, OriginalRo, Ro, Ub, i, j, 'L', 'W');
            FirstModifiedRBM.W[j + i * N_v] -= lr * grad.real();
            //std::cout << grad << "\n";
        }
    }
   
    //std::cout << "\nGrads for V (Lambda):\n";
    for (int i = 0; i < N_a; i++) {
        for (int j = 0; j < N_v; j++) {
            TComplex grad = GetGradLambdaMu(N, OriginalRo, Ro, Ub, i, j, 'L', 'V');
            FirstModifiedRBM.V[j + i * N_v] -= lr * grad.real();
            //std::cout << grad << "\n";
        }
    }

    //std::cout << "\nGrads for b (Lambda):\n";
    for (int i = 0; i < N_v; i++) {
        TComplex grad = GetGradLambdaMu(N, OriginalRo, Ro, Ub, i, 0, 'L', 'b');
        FirstModifiedRBM.b[i] -= lr * grad.real();
        //std::cout << grad << "\n";
    }

    //std::cout << "\nGrads for c (Lambda):\n";
    for (int i = 0; i < N_h; i++) {
        TComplex grad = GetGradLambdaMu(N, OriginalRo, Ro, Ub, i, 0, 'L', 'c');
        FirstModifiedRBM.c[i] -= lr * grad.real();
        //std::cout << grad << "\n";
    }

    //std::cout << "\nGrads for d (Lambda):\n";
    for (int i = 0; i < N_a; i++) {
        TComplex grad = GetGradLambdaMu(N, OriginalRo, Ro, Ub, i, 0, 'L', 'd');
        FirstModifiedRBM.d[i] -= lr * grad.real();
        //std::cout << grad << "\n";
    }

    //std::cout << "\nGrads for W (Mu):\n";
    for (int i = 0; i < N_h; i++) {
        for (int j = 0; j < N_v; j++) {
            TComplex grad = GetGradLambdaMu(N, OriginalRo, Ro, Ub, i, j, 'M', 'W');
            SecondModifiedRBM.W[j + i * N_v] -= lr * grad.imag();
            //std::cout << grad << "\n";
        }
    }

    //std::cout << "\nGrads for V (Mu):\n";
    for (int i = 0; i < N_a; i++) {
        for (int j = 0; j < N_v; j++) {
            TComplex grad = GetGradLambdaMu(N, OriginalRo, Ro, Ub, i, j, 'M', 'V');
            SecondModifiedRBM.V[j + i * N_v] -= lr * grad.real();
            //std::cout << grad << "\n";
        }
    }

    //std::cout << "\nGrads for b (Mu):\n";
    for (int i = 0; i < N_v; i++) {
        TComplex grad = GetGradLambdaMu(N, OriginalRo, Ro, Ub, i, 0, 'M', 'b');
        SecondModifiedRBM.b[i] -= lr * grad.imag();
        //std::cout << grad << "\n";
    }

    //std::cout << "\nGrads for c (Mu):\n";
    for (int i = 0; i < N_h; i++) {
        TComplex grad = GetGradLambdaMu(N, OriginalRo, Ro, Ub, i, 0, 'M', 'c');
        SecondModifiedRBM.c[i] -= lr * grad.imag();
        //std::cout << grad << "\n";
    }

    // Градиенты для d (Mu) равны нулю

    //std::cout << "\nGrads for d (Mu):\n";
    //for (int i = 0; i < N_a; i++) {
    //    SecondModifiedRBM.d[i];
    //    TComplex grad = GetGradLambdaMu(N, OriginalRo, Ro, Ub, i, 0, 'M', 'd');
    //    std::cout << grad << "\n";
    //}
    //std::cout << "\n\n";
}
