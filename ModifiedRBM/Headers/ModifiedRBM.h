#ifndef _MODIFIED_RBM_H_
#define _MODIFIED_RBM_H_

#include "DataType.h"
#include <string>

class ModifiedRBM {
public:
    int N_v, N_h, N_a;
    acc_number *W, *V, *b, *c, *d;

    ModifiedRBM(): N_v(0), N_h(0), N_a(0),
        W(nullptr), V(nullptr), b(nullptr), c(nullptr), d(nullptr) {};
    ~ModifiedRBM();

    void SetModifiedRBM(int _N_v, int _N_h, int _N_a, acc_number* _W, acc_number* _V,
        acc_number* _b, acc_number* _c, acc_number* _d);

    void PrintModifiedRBM(const std::string& NameRBM = "") const;
};

#endif //_MODIFIED_RBM_H_
