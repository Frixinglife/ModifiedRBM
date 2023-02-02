#ifndef _RO_MATRIX_EXPERIMENTS_H_
#define _RO_MATRIX_EXPERIMENTS_H_

#include "ComplexMKL.h"
#include "DataType.h"
#include <string>

void GetRoMatrixAndEig(int N_v, int N_h, int N_a, bool plot = false, std::string type = "random");
void GetWorkTime(int N_v, int N_h, int N_a, bool plot = false, std::string type = "random");

#endif //_RO_MATRIX_EXPERIMENTS_H_
