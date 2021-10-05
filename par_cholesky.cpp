#include "par_cholesky.h"

#include <algorithm>
#include <array>
#include <iostream>

#include <mkl.h>

namespace {
    //lvalue integer constants used for FORTRAN function calling (which needs pointers)
    constexpr int ONE = 1;
    constexpr int MINUS_ONE = -1;

    constexpr int nb_max = 32; //Value taken from reference-LAPACK, may require tuning.
    constexpr int ld_work_arr = nb_max + 1;

    //Converts from (zero-indexed) 2D index to a 1D index for *COLUMN MAJOR* storage.
    inline int to_flat_index(int nrows, int row, int col) { return nrows * col + row; }
}


//Implementation is essentially a direct translation from the FORTRAN dpbtrf from Reference LAPACK on Netlib:
//https://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_gad8b0e25cecc84ea3c5aa894ca1f1b5ca.html
int par_dpbtrf(int mat_dim, int bandwidth, double* ab, int ldab) {
    int status = 0;
    //Check input values
    if (mat_dim <= 0)
        return -2;
    if (bandwidth <= 0)
        return -3;
    if (ldab < (bandwidth + 1))
        return -5;

    int nb = ilaenv(&ONE, "DPBTRF", "L", &mat_dim, &bandwidth, &MINUS_ONE, &MINUS_ONE);
    std::array<double, ld_work_arr * nb_max> work_arr; //Temporary array used during computations
    std::fill(work_arr.begin(), work_arr.end(), 0.0);
    nb = std::min(nb, nb_max);

    for (int i = 0; i < mat_dim; i += nb) {
        std::cerr << "i: " << i << "\n";
        const int A11_width = std::min(nb, mat_dim - i);
        double* A11_start = ab + to_flat_index(ldab, 0, i);
        std::cerr << "Processing A11...\n";
        //Factorize A11 into U11
        status = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A11_width, A11_start, ldab - 1);
        if (status != 0)
            return status;

        if (i + A11_width < mat_dim) {
            const int A22_width = std::min(bandwidth - A11_width, mat_dim - i - A11_width);
            const int A33_width = std::min(A11_width, mat_dim - i - bandwidth);
            double* A21_start = ab + to_flat_index(ldab, A11_width, i);
            double* A22_start = ab + to_flat_index(ldab, 0, i + A11_width);

            if (A22_width > 0) {
                std::cerr << "Processing A21...\n";
                cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                            A22_width, A11_width, 1.0, A11_start, ldab - 1, A21_start, ldab - 1);

                std::cerr << "Processing A22...\n";
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            A22_width, A11_width, -1.0, A21_start,
                            ldab - 1, 1.0, A22_start, ldab - 1);
            }

            if (A33_width > 0) {
                double* A32_start = ab + to_flat_index(ldab, bandwidth - A11_width, i + A11_width);
                double* A33_start = ab + to_flat_index(ldab, 0, i + bandwidth);
                //Copy the upper triangle of the A31 block into the temporary work array
                std::cerr << "Copying A31...\n";
                for (int j = 0; j < A11_width; ++j) {
                    for (int k = 0; k < std::min(j, A33_width); ++k) {
                        work_arr[to_flat_index(ld_work_arr, j, k)] =
                            *(ab + to_flat_index(ldab, bandwidth - j + k, j + i));
                    }
                }

                std::cerr << "Processing A31...\n";
                cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                            A33_width, A11_width, 1.0, A11_start, ldab - 1, &work_arr[0], ld_work_arr);

                if (A22_width > 0) {
                    std::cerr << "Processing A32...\n";
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                A33_width, A22_width, A11_width,
                                -1.0, &work_arr[0], ld_work_arr,
                                A21_start, ldab-1, 1.0,
                                A32_start, ldab-1);
                }
                std::cerr << "Processing A33...\n";
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            A33_width, A11_width, -1.0, &work_arr[0], ld_work_arr,
                            1.0, A33_start, ldab-1);

                std::cerr << "Copying A31 back...\n";
                //Copy back from work array to A31 upper triangle.
                for (int j = 0; j < A11_width; ++j) {
                    for (int k = 0; k < std::min(j, A33_width); ++k) {
                        *(ab + to_flat_index(ldab, bandwidth - j + k, j + i)) =
                            work_arr[to_flat_index(nb_max + 1, j, k)];
                    }
                }
            }
        }
    }

    return 0;
}