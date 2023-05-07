#ifndef _MODIFIED_RBM_H_
#define _MODIFIED_RBM_H_

#include "DataType.h"
#include <string>

class ModifiedRBM {
public:
    int N_v, N_h, N_a;
    acc_number *W_l, *W_m, *V_l, *V_m, *b_l, *b_m, *c_l, *c_m, *d;

    ModifiedRBM(): N_v(0), N_h(0), N_a(0), W_l(nullptr), W_m(nullptr), V_l(nullptr), V_m(nullptr), 
        b_l(nullptr), b_m(nullptr), c_l(nullptr), c_m(nullptr), d(nullptr) {};
    ~ModifiedRBM();

    void SetModifiedRBM(int _N_v, int _N_h, int _N_a, acc_number* _W_l, acc_number* _W_m, acc_number* _V_l,
        acc_number* _V_m, acc_number* _b_l, acc_number* _b_m, acc_number* _c_l, acc_number* _c_m, acc_number* _d);

    void PrintModifiedRBM() const;
};

#endif //_MODIFIED_RBM_H_
