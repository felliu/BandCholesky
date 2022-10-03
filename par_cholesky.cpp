#include "par_cholesky.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#ifdef USE_MKL_
#include <mkl.h>
#else
#ifdef USE_BLIS
#include <blis/blis.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#endif

#if defined USE_PLASMA_ || defined USE_BLIS
#include "lapack_wrapper.h"
#endif

namespace {
    //lvalue integer constants used for FORTRAN function calling (which needs pointers)
    constexpr int ONE = 1;
    constexpr int MINUS_ONE = -1;

    constexpr int nb_max = 75; //Value taken from reference-LAPACK, may require tuning.
    //constexpr int ld_work_arr = nb_max + 1;

    //Converts from (zero-indexed) 2D index to a 1D index for *COLUMN MAJOR* storage.
    inline int to_flat_index(int nrows, int row, int col) { return nrows * col + row; }
    inline int to_flat_index2(int nrows, int row, int col) { return nrows * (col - 1) + (row - 1); }
#ifdef USE_BLIS
    double MINUS_ONE_D = -1.0;
    double ONE_D = 1.0;
#endif
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

    const int nb = bandwidth / 2;
    const int ld_work_arr = nb + 1;
    std::vector<double> work_arr(nb * ld_work_arr); //Temporary array used during computations
#pragma omp parallel num_threads(7) shared(nb, work_arr)
{
#pragma omp single nowait
{
    double* A22_start = nullptr;
    double* A32_start = nullptr;
    double* A33_start = nullptr;
    //Start of the chain:
    #pragma omp task depend(out:A22_start,A32_start,A33_start)
    {}
    for (int i = 0; i < mat_dim; i += nb) {
        const int A11_width = std::min(nb, mat_dim - i);
        double* A11_start = ab + to_flat_index(ldab, 0, i);
        A22_start = nullptr;
        A32_start = nullptr;
        A33_start = nullptr;
        //Factorize A11 into U11
        #pragma omp task depend(in:A22_start) depend(out:A11_start)
#ifdef USE_BLIS
        status = dpotrf_wrapper(LAPACK_COL_MAJOR, 'L', A11_width, A11_start, ldab - 1);
#else
        status = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A11_width, A11_start, ldab - 1);
#endif
        /*if (status != 0)
            return status;
        */
        if (i + A11_width < mat_dim) {
            const int A22_width = std::min(bandwidth - A11_width, mat_dim - i - A11_width);
            const int A33_width = std::min(A11_width, mat_dim - i - bandwidth);
            double* A21_start = ab + to_flat_index(ldab, A11_width, i);
            A22_start = ab + to_flat_index(ldab, 0, i + A11_width);

            if (A22_width > 0) {
                #pragma omp task depend(in:A11_start,A32_start) depend(out:A21_start) \
                        firstprivate(A22_width, A11_width, A11_start, ldab, A21_start)
                //TODO
                /*bli_dtrsm(BLIS_RIGHT, BLIS_LOWER, BLIS_NO_TRANSPOSE, BLIS_NONUNIT_DIAG,
                          A22_width, A11_width, &ONE, A11_start, 1, ldab - 1)*/
                cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                            A22_width, A11_width, 1.0, A11_start, ldab - 1, A21_start, ldab - 1);

                #pragma omp task depend(in:A21_start,A33_start) depend(out:A22_start) \
                        firstprivate(A22_width, A11_width, A21_start, ldab, A22_start)
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            A22_width, A11_width, -1.0, A21_start,
                            ldab - 1, 1.0, A22_start, ldab - 1);
            }

            if (A33_width > 0) {
                A32_start = ab + to_flat_index(ldab, bandwidth - A11_width, i + A11_width);
                A33_start = ab + to_flat_index(ldab, 0, i + bandwidth);
                #pragma omp task depend(inout:work_arr) \
                        firstprivate(ab, ld_work_arr, A11_width, A33_width, bandwidth, ldab)
                {
                //Copy the upper triangle of the A31 block into the temporary work array
#ifdef USE_MKL_
                    mkl_domatcopy('C', 'N', A33_width, A11_width, 1.0,
                                  ab, A33_width, &work_arr[0], A33_width);
#else
                    for (int j = 0; j < A11_width; ++j) {
                        const int k_max = std::min(j + 1, A33_width);
                        for (int k = 0; k < k_max; ++k) {
                            work_arr[to_flat_index(ld_work_arr, k, j)] =
                                *(ab + to_flat_index(ldab, bandwidth - j + k, j + i));
                        }
                    }
#endif
                }

                #pragma omp task depend(in:A11_start) depend(inout:work_arr) \
                        firstprivate(A33_width, A11_width, A11_start, ldab, ld_work_arr)
                cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                            A33_width, A11_width, 1.0, A11_start, ldab - 1, &work_arr[0], ld_work_arr);

                if (A22_width > 0)
                    #pragma omp task depend(in:A21_start,work_arr) depend(out:A32_start) \
                            firstprivate(A33_width, A22_width, A11_width, ld_work_arr, A21_start, ldab, A32_start)
#ifdef USE_BLIS
                    bli_dtrmm3(BLIS_LEFT, BLIS_UPPER, BLIS_NO_TRANSPOSE,
                               BLIS_NONUNIT_DIAG, BLIS_TRANSPOSE,
                               A33_width, A22_width, &MINUS_ONE_D,
                               &work_arr[0], 1, ld_work_arr,
                               A21_start, 1, ldab - 1,
                               &ONE_D, A32_start, 1, ldab - 1);
#else
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                A33_width, A22_width, A11_width,
                                -1.0, &work_arr[0], ld_work_arr,
                                A21_start, ldab - 1, 1.0,
                                A32_start, ldab - 1);
#endif

                #pragma omp task depend(in:work_arr) depend(out:A33_start) \
                        firstprivate(A33_width, A11_width, A33_start, ldab, ld_work_arr)
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            A33_width, A11_width, -1.0, &work_arr[0], ld_work_arr,
                            1.0, A33_start, ldab - 1);

                #pragma omp task depend(inout:work_arr) \
                        firstprivate(ab, ld_work_arr, A11_width, A33_width, bandwidth, ldab)
                {
                    //Copy back from work array to A31 upper triangle.
#ifdef USE_MKL_
                    mkl_domatcopy('C', 'N', A33_width, A11_width, 1.0,
                                  &work_arr[0], A11_width, ab, A11_width);
#else
                    for (int j = 0; j < A11_width; ++j) {
                        const int k_max = std::min(j + 1, A33_width);
                        for (int k = 0; k < k_max; ++k) {
                            *(ab + to_flat_index(ldab, bandwidth - j + k, j + i)) =
                                work_arr[to_flat_index(ld_work_arr, k, j)];
                        }
                    }
#endif
                }
            }
        }
    }

} //End OMP single
} //End OMP parallel
return status;
}

//Implementation is essentially a direct translation from the FORTRAN dpbtrf from Reference LAPACK on Netlib:
//https://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_gad8b0e25cecc84ea3c5aa894ca1f1b5ca.html
/*int par_dpbtrf_barrier(int mat_dim, int bandwidth, double* ab, int ldab) {
    int status = 0;
    //Check input values
    if (mat_dim <= 0)
        return -2;
    if (bandwidth <= 0)
        return -3;
    if (ldab < (bandwidth + 1))
        return -5;

    const int nb = bandwidth / 2;
    const int ld_work_arr = nb + 1;
    std::vector<double> work_arr(nb * ld_work_arr); //Temporary array used during computations
#pragma omp parallel num_threads(7) shared(nb, work_arr)
{
#pragma omp single nowait
{
    for (int i = 0; i < mat_dim; i += nb) {
        const int A11_width = std::min(nb, mat_dim - i);
        double* A11_start = ab + to_flat_index(ldab, 0, i);

        //Factorize A11 into U11
        #pragma omp task depend(out:A11_start)
        status = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A11_width, A11_start, ldab - 1);

        if (i + A11_width < mat_dim) {
            const int A22_width = std::min(bandwidth - A11_width, mat_dim - i - A11_width);
            const int A33_width = std::min(A11_width, mat_dim - i - bandwidth);
            double* A21_start = ab + to_flat_index(ldab, A11_width, i);
            double* A22_start = ab + to_flat_index(ldab, 0, i + A11_width);

            if (A22_width > 0) {
                #pragma omp task depend(in:A11_start) depend(out:A21_start)
                cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                            A22_width, A11_width, 1.0, A11_start, ldab - 1, A21_start, ldab - 1);

                #pragma omp task depend(in:A21_start)
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            A22_width, A11_width, -1.0, A21_start,
                            ldab - 1, 1.0, A22_start, ldab - 1);
            }

            if (A33_width > 0) {
                double* A32_start = ab + to_flat_index(ldab, bandwidth - A11_width, i + A11_width);
                double* A33_start = ab + to_flat_index(ldab, 0, i + bandwidth);
                #pragma omp task depend(out:work_arr)
                {
                //Copy the upper triangle of the A31 block into the temporary work array
                for (int j = 0; j < A11_width; ++j) {
                    const int k_max = std::min(j + 1, A33_width);
                    for (int k = 0; k < k_max; ++k) {
                        work_arr[to_flat_index(ld_work_arr, k, j)] =
                            *(ab + to_flat_index(ldab, bandwidth - j + k, j + i));
                    }
                }
                }

                #pragma omp task depend(in:A11_start) depend(inout:work_arr)
                cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                            A33_width, A11_width, 1.0, A11_start, ldab - 1, &work_arr[0], ld_work_arr);

                if (A22_width > 0)
                    #pragma omp task depend(in:A21_start,work_arr)
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                A33_width, A22_width, A11_width,
                                -1.0, &work_arr[0], ld_work_arr,
                                A21_start, ldab - 1, 1.0,
                                A32_start, ldab - 1);

                #pragma omp task depend(in:work_arr)
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                            A33_width, A11_width, -1.0, &work_arr[0], ld_work_arr,
                            1.0, A33_start, ldab - 1);

                #pragma omp task depend(in:work_arr)
                {
                //Copy back from work array to A31 upper triangle.
                for (int j = 0; j < A11_width; ++j) {
                    const int k_max = std::min(j + 1, A33_width);
                    for (int k = 0; k < k_max; ++k) {
                        *(ab + to_flat_index(ldab, bandwidth - j + k, j + i)) =
                            work_arr[to_flat_index(ld_work_arr, k, j)];
                    }
                }
                }
            }
        }
        #pragma omp taskwait
    }

} //End OMP single
} //End OMP parallel
return status;
}*/
