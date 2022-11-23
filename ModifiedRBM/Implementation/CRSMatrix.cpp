#include "CRSMatrix.h"
#include <iostream>
#include <iomanip>

CRSMatrix::CRSMatrix(int _n, int _nz) {
    n = _n;
    nz = _nz;

    if (n != 0 && nz != 0) {
        val = new MKL_Complex16[nz];
        colIndex = new int[nz];
        rowPtr = new int[n + 1];
    }
}

CRSMatrix::CRSMatrix(int _n, int _nz, MKL_Complex16* _val, int* _colIndex, int* _rowPtr) {
	n = _n;
	nz = _nz;
	val = new MKL_Complex16[nz];
	colIndex = new int[nz];
	rowPtr = new int[n + 1];

	for (int i = 0; i < nz; i++) {
		val[i] = _val[i];
		colIndex[i] = _colIndex[i];
	}

	for (int i = 0; i < n + 1; i++) {
		rowPtr[i] = _rowPtr[i];
	}
}

CRSMatrix::CRSMatrix(int _n, MKL_Complex16* matrix) {
    n = _n;
    double eps = 1e-7;

    nz = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            MKL_Complex16 elem = matrix[j + i * n];
            if (abs(elem.real()) < eps && abs(elem.imag()) < eps) {
                continue;
            }
            else {
                nz++;
            }
        }
    }

    val = new MKL_Complex16[nz];
    colIndex = new int[nz];
    rowPtr = new int[n + 1];

    rowPtr[0] = 0;
    int k = 0;

    for (int i = 0; i < n; i++) {
        int count = rowPtr[i];
        for (int j = 0; j < n; j++) {
            MKL_Complex16 elem = matrix[j + i * n];
            if (abs(elem.real()) < eps && abs(elem.imag()) < eps) {
                continue;
            }
            else {
                count++;
                val[k] = elem;
                colIndex[k] = j;
                k++;
            }
        }
        rowPtr[i + 1] = count;
    }
}

CRSMatrix::CRSMatrix(const CRSMatrix& matrix) {
    n = matrix.n;
    nz = matrix.nz;
    val = new MKL_Complex16[nz];
    colIndex = new int[nz];
    rowPtr = new int[n + 1];

    for (int i = 0; i < nz; i++) {
        val[i] = matrix.val[i];
        colIndex[i] = matrix.colIndex[i];
    }

    for (int i = 0; i < n + 1; i++) {
        rowPtr[i] = matrix.rowPtr[i];
    }
}

CRSMatrix& CRSMatrix::operator=(const CRSMatrix& matrix) {
    if (this == &matrix) {
        return *this;
    }

    n = matrix.n;
    nz = matrix.nz;
    val = new MKL_Complex16[nz];
    colIndex = new int[nz];
    rowPtr = new int[n + 1];

    for (int i = 0; i < nz; i++) {
        val[i] = matrix.val[i];
        colIndex[i] = matrix.colIndex[i];
    }

    for (int i = 0; i < n + 1; i++) {
        rowPtr[i] = matrix.rowPtr[i];
    }

    return *this;
}

CRSMatrix::~CRSMatrix() {
    if (n != 0 && nz != 0) {
        delete[] val;
        delete[] colIndex;
        delete[] rowPtr;
    }
}

void CRSMatrix::PrintCRS(std::string name) {
    std::cout << name << ":\n";
    std::cout << "N = " << n << ", NZ = " << nz << ":\n";

    std::cout << "Values: ";
    for (int i = 0; i < nz; i++) {
        std::cout << val[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Cols: ";
    for (int i = 0; i < nz; i++) {
        std::cout << colIndex[i] << " ";
    }
    std::cout << "\n";

    std::cout << "RowIndexes: ";
    for (int i = 0; i < n + 1; i++) {
        std::cout << rowPtr[i] << " ";
    }
    std::cout << "\n";

    MKL_Complex16* Intermed = new MKL_Complex16[n * n];
    for (int i = 0; i < n * n; i++) {
        Intermed[i] = MKL_Complex16(0.0, 0.0);
    }

    for (int i = 0; i < n; i++) {
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
            Intermed[colIndex[j] + i * n] = val[j];
        }
    }

    std::cout << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setw(30) << Intermed[j + i * n];
        }
        std::cout << "\n";
    }
    std::cout << "\n\n";

    delete[] Intermed;
}

CRSMatrix CRSMatrix::GetHermitianConjugateCRS() {
    CRSMatrix Result(n, nz);

    int* rowPtrTemp = new int[n + 2];

	for (int i = 0; i < nz; i++) {
		Result.colIndex[i] = 0;
		Result.val[i] = MKL_Complex16(0.0, 0.0);
	}

	for (int i = 0; i < n + 2; i++) {
		rowPtrTemp[i] = 0;
	}

	for (int i = 0; i < nz; i++) {
		int ind = colIndex[i] + 2;
		rowPtrTemp[ind] += 1;
	}

	for (int i = 2; i < n + 2; i++) {
		rowPtrTemp[i] += rowPtrTemp[i - 1];
	}

	for (int i = 0; i < n; i++) {
		for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
			int ind_1 = colIndex[j] + 1;
			int ind_2 = rowPtrTemp[ind_1];
			Result.colIndex[ind_2] = i;
			Result.val[ind_2] = MKL_Complex16(val[j].real(), -val[j].imag());
            rowPtrTemp[ind_1] += 1;
		}
	}

    for (int i = 0; i < n + 1; i++) {
        Result.rowPtr[i] = rowPtrTemp[i];
    }

    return Result;
}

MKL_Complex16* CRSMatrix::MultCRSDense(CRSMatrix CRS_matrix, MKL_Complex16* Dense_matrix, int n) {
    MKL_Complex16* Result = new MKL_Complex16[n * n];

    for (int i = 0; i < n; i++) {
	    for (int j = CRS_matrix.rowPtr[i]; j < CRS_matrix.rowPtr[i + 1]; j++) {
            int col = CRS_matrix.colIndex[j];
            MKL_Complex16 CSR_val = CRS_matrix.val[j];
		    for (int k = 0; k < n; k++) {
                MKL_Complex16 Dense_val = Dense_matrix[k + col * n];
			    Result[k + i * n] += CSR_val * Dense_val;
		    }
	    }
    }

    return Result;
}

MKL_Complex16* CRSMatrix::MultDenseCSR(MKL_Complex16* Dense_matrix, CRSMatrix CRS_matrix, int n) {
    MKL_Complex16* Result = new MKL_Complex16[n * n];
    CRSMatrix TransCRS = CRS_matrix.GetHermitianConjugateCRS();

    // A * B_CSR = (B_CSR^T * A^T)^T

    for (int i = 0; i < n; i++) {
        for (int j = TransCRS.rowPtr[i]; j < TransCRS.rowPtr[i + 1]; j++) {
            int col = TransCRS.colIndex[j];
            MKL_Complex16 CSR_val = TransCRS.val[j];
            for (int k = 0; k < n; k++) {
                MKL_Complex16 Dense_val = Dense_matrix[col + k * n]; // A^T
                Result[i + k * n] += CSR_val * Dense_val;  // Res^T
            }
        }
    }

    return Result;
}
