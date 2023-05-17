#include "par_cholesky.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
#include "PB_matrix.h"

#ifdef USE_MKL_
#include <mkl.h>
#else
#ifdef USE_BLIS_
#include <blis/blis.h>
#include <blis/cblas.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
#endif

#include <omp.h>

#if defined USE_PLASMA_ || defined USE_BLIS_
#include "lapack_wrapper.h"
#endif

namespace {
    //lvalue integer constants used for FORTRAN function calling (which needs pointers)
    constexpr int ONE = 1;
    constexpr int MINUS_ONE = -1;

    constexpr int MAX_LEVELS = 32;

    constexpr int nb_max = 75; //Value taken from reference-LAPACK, may require tuning.
    //constexpr int ld_work_arr = nb_max + 1;

    //Converts from (zero-indexed) 2D index to a 1D index for *COLUMN MAJOR* storage.
    inline int to_flat_index(int nrows, int row, int col) { return nrows * col + row; }
    inline int to_flat_index2(int nrows, int row, int col) { return nrows * (col - 1) + (row - 1); }

    //Function to select a target block size for parallelization of a given matrix.
    //Very crudely, the function tries to set the number of blocks to divide each window to
    //satisfy the following:
    //1) n- 1 should divide bandwidth
    //2) n is such that the block size is approximately 50 by 50
    //3) n is not greater than the number of physical cores on the system
    //   (as reported by omp_get_max_threads)
    int calc_num_sub_blocks(int bandwidth) {
        constexpr int target_block_sz = 50;
        int levels = 0;
        //We want the block sizes of the sub-blocks to be reasonable
        //while still keeping that the bandwidth is divisible by the number of levels - 1,
        //so that the sub-blocks line up properly
        if (bandwidth <= 2 * target_block_sz) {
            //We assume that the bandwidth is, at the very least, even.
            assert(bandwidth % 2 == 0);
            return 3;
        }

        int num_threads = omp_get_max_threads();

        levels = std::min((bandwidth + target_block_sz) / target_block_sz, num_threads);
        //We need to select the number of levels so that (levels - 1) divides the bandwidth.
        //If we have fewer levels than the number of threads, we scan upwards for an appropriate
        //number of levels (such that (levels - 1) divides bandwidth).
        if (levels < num_threads) {
            while (levels <= num_threads && bandwidth % (levels - 1) != 0)
                ++levels;
            //Check if we found an appropriate number of levels
            if (bandwidth % (levels - 1) == 0)
                return levels;
        }
        //Else, the current value of levels is the number of threads, and we scan downwards for
        //suitable values
        else {
            while (levels >= 3 && bandwidth % (levels - 1) != 0)
                --levels;

            if (bandwidth % (levels - 1) == 0)
                return levels;
        }
        //If we still haven't found a suitable value,
        //accept the performance loss and just set the number of levels to three,
        //with the assumption that the bandwidth is even
        assert(bandwidth % 2 == 0);
        return 3;
    }
#ifdef USE_BLIS_
    double MINUS_ONE_D = -1.0;
    double ONE_D = 1.0;
#endif
}

int par_fine_dpbtrf(int mat_dim, int bandwidth, double* ab, int ldab) {
#ifdef USE_MKL_
    LAPACKE_set_nancheck(0);
#endif
    const int levels = calc_num_sub_blocks(bandwidth);
    const int threads = std::min(2*levels, omp_get_max_threads());
    //Somewhat ugly solution for now, in the future should modify the sub block
    //calculation to account for this...
    if (levels > MAX_LEVELS)
        return -1;
    const int nb = bandwidth / (levels - 1);
    const int ld_work_arr = nb + 1;
    std::vector<double> work_arr(nb * ld_work_arr);

    //Dummy array used to specify the task dependencies.
    //entry (i, j) in the array logically represents the sub-block
    //at index (i, j) in the grid.
    char task_dep[MAX_LEVELS][MAX_LEVELS];
#pragma omp parallel num_threads(levels)
#pragma omp single
{
    for (int i = 0; i < mat_dim; i += nb) {
        const int A11_width = std::min(nb, mat_dim - i);
        double* A11_start = ab + to_flat_index(ldab, 0, i);
        #pragma omp task depend(out:task_dep[0][0]) depend(in: task_dep[1][1])
#ifdef USE_BLIS_
        dpotrf_wrapper(LAPACK_COL_MAJOR, 'L', A11_width, A11_start, ldab - 1);
#else
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A11_width, A11_start, ldab - 1);
#endif
        for (int block_i = 1; block_i < levels - 1; ++block_i) {
            //Stop processing once we hit the bottom of the matrix
            const int nrows = std::min(nb, mat_dim - i - block_i * nb);
            //Check to see if the next sub-block is past the bottom of the matrix
            if (nrows <= 0)
                break;

            double* Ai1_start = ab + to_flat_index(ldab, block_i * nb, i);
            #pragma omp task depend(in:task_dep[0][0], task_dep[block_i + 1][1]) \
                             depend(out: task_dep[block_i][0])
            cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                        CblasNonUnit, nrows, A11_width, 1.0, A11_start, ldab - 1,
                        Ai1_start, ldab - 1);
            for (int block_j = 1; block_j <= block_i - 1; ++block_j) {
                double* A_ij = ab + to_flat_index(ldab, block_i * nb - block_j * nb, i + block_j * nb);
                double* L_j1 = ab + to_flat_index(ldab, block_j * nb, i);
                const int m = nrows;
                const int n = nb;
                const int k = nb;
                #pragma omp task depend(in:task_dep[block_i][0], task_dep[block_j][0]) \
                                 depend(out: task_dep[block_i][block_j])
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            m, n, k, -1.0,
                            Ai1_start, ldab - 1,
                            L_j1, ldab - 1, 1.0,
                            A_ij, ldab - 1);
            }

            double* Aii_start = ab + to_flat_index(ldab, 0, i + block_i * nb);
            #pragma omp task depend(in: task_dep[block_i][0]) \
                             depend(in: task_dep[block_i + 1][block_i + 1]) \
                             depend(out: task_dep[block_i][block_i])
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                        nrows, A11_width, -1.0, Ai1_start, ldab - 1,
                        1.0, Aii_start, ldab - 1);
        }

        const int nrows_bottom = std::min(A11_width, mat_dim - i - bandwidth);

        //Check if we have any blocks left, these need special handling since the bottom
        //left block is upper triangular (its lower left triangle lies outside the band).
        if (nrows_bottom <= 0)
            continue;

        const int block_i = levels - 1;
        double* bottom_left_start = ab + to_flat_index(ldab, block_i * nb, i);
        //Copy upper triangle of the bottom left block into the work array
        #pragma omp task depend(in: task_dep[block_i][0]) depend(out: work_arr)
        {
            for (int j = 0; j < nb; ++j) {
                const int k_max = std::min(j + 1, nrows_bottom);
                for (int k = 0; k < k_max; ++k) {
                    work_arr[to_flat_index(ld_work_arr, k, j)] =
                        *(bottom_left_start + to_flat_index(ldab, k - j, j));
                }
            }
        }

        #pragma omp task depend(in: task_dep[0][0]) depend(inout: work_arr)
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    nrows_bottom, A11_width, 1.0, A11_start, ldab - 1,
                    &work_arr[0], ld_work_arr);

        for (int block_j = 1; block_j <= block_i - 1; ++block_j) {
            double* A_ij = ab + to_flat_index(ldab, block_i * nb - block_j * nb, i + block_j * nb);
            double* L_j1 = ab + to_flat_index(ldab, block_j * nb, i);
            const int m = nrows_bottom;
            const int n = nb;
            const int k = nb;
            #pragma omp task depend(in: task_dep[block_j][0], work_arr) \
                             depend(out: task_dep[block_i][block_j])
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        m, n, k, -1.0,
                        &work_arr[0], ld_work_arr,
                        L_j1, ldab - 1, 1.0,
                        A_ij, ldab - 1);
        }

        double* diag_start = ab + to_flat_index(ldab, 0, i + block_i * nb);
        #pragma omp task depend(in: work_arr) depend(out: task_dep[block_i][block_i])
        cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                    nrows_bottom, nb, -1.0, &work_arr[0], ld_work_arr,
                    1.0, diag_start, ldab - 1);

        #pragma omp task depend(in: work_arr) depend(out: task_dep[block_i][0])
        {
            //Copy upper triangle of bottom left block back into main matrix
            for (int j = 0; j < nb; ++j) {
                const int k_max = std::min(j + 1, nrows_bottom);
                for (int k = 0; k < k_max; ++k) {
                    *(bottom_left_start + to_flat_index(ldab, k - j, j)) =
                        work_arr[to_flat_index(ld_work_arr, k, j)];
                }
            }
        }
    }
}
    return 0;
}


