#include "ModifiedRBM.h"
#include <iostream>
#include <iomanip>

ModifiedRBM::~ModifiedRBM() {
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

void ModifiedRBM::SetModifiedRBM(int _N_v, int _N_h, int _N_a, acc_number* _W_l, acc_number* _W_m, acc_number* _V_l, 
    acc_number* _V_m, acc_number* _b_l, acc_number* _b_m, acc_number* _c_l, acc_number* _c_m, acc_number* _d) {

    N_v = _N_v;
    N_h = _N_h;
    N_a = _N_a;

    const int N_h_N_v = N_h * N_v;
    const int N_a_N_v = N_a * N_v;

    W_l = new acc_number[N_h_N_v];
    W_m = new acc_number[N_h_N_v];
    V_l = new acc_number[N_a_N_v];
    V_m = new acc_number[N_a_N_v];
    b_l = new acc_number[N_v];
    b_m = new acc_number[N_v];
    c_l = new acc_number[N_h];
    c_m = new acc_number[N_h];
    d = new acc_number[N_a];

    for (int i = 0; i < N_h_N_v; i++) {
        W_l[i] = _W_l[i];
        W_m[i] = _W_m[i];
    }

    for (int i = 0; i < N_a_N_v; i++) {
        V_l[i] = _V_l[i];
        V_m[i] = _V_m[i];
    }

    for (int i = 0; i < N_v; i++) {
        b_l[i] = _b_l[i];
        b_m[i] = _b_m[i];
    }

    for (int i = 0; i < N_h; i++) {
        c_l[i] = _c_l[i];
        c_m[i] = _c_m[i];
    }

    for (int i = 0; i < N_a; i++) {
        d[i] = _d[i];
    }
}

void ModifiedRBM::PrintModifiedRBM() const {
    std::cout << "N_v = " << N_v << ", N_h = " << N_h << ", N_a = " << N_a << "\n\n";

    std::cout << "W_lambda:" << "\n";
    for (int i = 0; i < N_h; i++) {
        for (int j = 0; j < N_v; j++) {
            std::cout << std::setw(15) << W_l[j + i * N_v];
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "W_mu:" << "\n";
    for (int i = 0; i < N_h; i++) {
        for (int j = 0; j < N_v; j++) {
            std::cout << std::setw(15) << W_m[j + i * N_v];
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "V_lambda:" << "\n";
    for (int i = 0; i < N_a; i++) {
        for (int j = 0; j < N_v; j++) {
            std::cout << std::setw(15) << V_l[j + i * N_v];
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "V_mu:" << "\n";
    for (int i = 0; i < N_a; i++) {
        for (int j = 0; j < N_v; j++) {
            std::cout << std::setw(15) << V_m[j + i * N_v];
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "b_lambda:" << "\n";
    for (int i = 0; i < N_v; i++) {
        std::cout << b_l[i] << "\n";
    }
    std::cout << "\n";

    std::cout << "b_mu:" << "\n";
    for (int i = 0; i < N_v; i++) {
        std::cout << b_m[i] << "\n";
    }
    std::cout << "\n";

    std::cout << "c_lambda:" << "\n";
    for (int i = 0; i < N_h; i++) {
        std::cout << c_l[i] << "\n";
    }
    std::cout << "\n";

    std::cout << "c_mu:" << "\n";
    for (int i = 0; i < N_h; i++) {
        std::cout << c_m[i] << "\n";
    }
    std::cout << "\n";

    std::cout << "d:" << "\n";
    for (int i = 0; i < N_a; i++) {
        std::cout << d[i] << "\n";
    }
    std::cout << "\n";
}
