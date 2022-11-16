#ifndef _RO_MATRIX_EXPERIMENTS_H_
#define _RO_MATRIX_EXPERIMENTS_H_

#include "ComplexMKL.h"
#include "DataType.h"

void GetRoMatrixAndEig(int N_v, int N_h, int N_a, bool plot = false);
void GetWorkTime(int N_v, int N_h, int N_a, bool plot = false);

#endif //_RO_MATRIX_EXPERIMENTS_H_
