#ifndef RUN_BENCHMARKS_H
#define RUN_BENCHMARKS_H

#include "PB_matrix.h"
#include <mkl.h>

template <typename T>
double run_parallel_benchmark(int N_tries, PB_matrix<T>& mat) {
    double total_time = 0.0;
    int status = 0;
    //Keep a copy of the original matrix for re-use, since dpbtrf modifies the matrix in-place.
    PB_matrix<T> mat_cpy(mat);
    //Warm-start
    status = par_dpbtrf(static_cast<int>(mat.size),
                        static_cast<int>(mat.bandwidth),
                        &mat.data[0],
                        static_cast<int>(mat.bandwidth + 1));

    for (int i = 0; i < N_tries; ++i) {
        const double start = dsecnd();
        status = par_dpbtrf(static_cast<int>(mat.size),
                            static_cast<int>(mat.bandwidth),
                            &mat.data[0],
                            static_cast<int>(mat.bandwidth + 1));
        const double end = dsecnd();
        total_time += (end - start);
        mat = mat_cpy;
    }

    mat.factorized = true;
    return total_time;
}

template <typename T>
double run_MKL_benchmark(int N_tries, PB_matrix<T>& mat) {
    double total_time = 0.0;
    int status = 0;
    //Keep a copy of the original matrix for re-use, since dpbtrf modifies the matrix in-place.
    PB_matrix<T> mat_cpy(mat);
    //Warm-start
    status = LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'L',
                            static_cast<int>(mat.size),
                            static_cast<int>(mat.bandwidth),
                            &mat.data[0],
                            static_cast<int>(mat.bandwidth + 1));
    assert(status == 0);

    for (int i = 0; i < N_tries; ++i) {
        mat = mat_cpy;
        const double start = dsecnd();
        status = LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'L',
                                static_cast<int>(mat.size),
                                static_cast<int>(mat.bandwidth),
                                &mat.data[0],
                                static_cast<int>(mat.bandwidth + 1));
        const double end = dsecnd();
        total_time += (end - start);
        assert(status == 0);
    }

    mat.factorized = true;
    return total_time;
}

#ifdef USE_PLASMA
template <typename T>
double run_plasma_benchmark(int N_tries, PB_matrix<T>& mat) {
    double total_time = 0.0;
    int status = 0;
    const plasma_enum_t uplo_plasma = plasma_uplo_const('L');
    //Keep a copy of the original matrix for re-use, since dpbtrf modifies the matrix in-place.
    PB_matrix<T> mat_cpy(mat);
    //Warm-start
    status = plasma_dpbtrf(uplo_plasma,
                           static_cast<int>(mat.size),
                           static_cast<int>(mat.bandwidth),
                           &mat.data[0],
                           static_cast<int>(mat.bandwidth + 1));

    for (int i = 0; i < N_tries; ++i) {
        const double start = dsecnd();
        status = plasma_dpbtrf(uplo_plasma,
                               static_cast<int>(mat.size),
                               static_cast<int>(mat.bandwidth),
                               &mat.data[0],
                               static_cast<int>(mat.bandwidth + 1));
        const double end = dsecnd();
        total_time += (end - start);
        mat = mat_cpy;
    }

    mat.factorized = true;
    return total_time;
}
#endif


#endif
