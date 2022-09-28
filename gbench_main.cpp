#include "PB_matrix.h"
#include "matrix_generator.h"
#include "par_cholesky.h"
#ifdef USE_MKL_
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <benchmark/benchmark.h>

static void custom_args(benchmark::internal::Benchmark* b) {
    constexpr int low_range_min = 10;
    constexpr int low_range_max = 100;
    constexpr int low_range_step = 10;

    constexpr int high_range_min = 100;
    constexpr int high_range_max = 2000;
    constexpr int high_range_step = 50;
    for (int i = low_range_min; i < low_range_max; i += low_range_step) {
        b->Args({i});
    }
    for (int i = high_range_min; i <= high_range_max; i += high_range_step) {
        b->Args({i});
    }
}

static constexpr size_t default_dim = 100000;

static void BM_Lapacke(benchmark::State& state) {
    PB_matrix<double> mat = get_random_pd_bandmat<double>(default_dim, state.range(0));
    PB_matrix<double> mat_cpy = mat;
#ifdef USE_MKL_
    LAPACKE_set_nancheck(0);
#endif
    for (auto _: state) {
        state.PauseTiming();
        mat = mat_cpy;
        state.ResumeTiming();
        LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'L',
                       static_cast<int>(mat.size),
                       static_cast<int>(mat.bandwidth),
                       &mat.data[0],
                       static_cast<int>(mat.bandwidth + 1));
    }
}
BENCHMARK(BM_Lapacke)->Iterations(100)->Apply(custom_args);

static void PM_par_pbtrf(benchmark::State& state) {
    PB_matrix<double> mat = get_random_pd_bandmat<double>(default_dim, state.range(0));
    PB_matrix<double> mat_cpy = mat;

    for (auto _: state) {
        state.PauseTiming();
        mat = mat_cpy;
        state.ResumeTiming();
        par_dpbtrf(static_cast<int>(mat.size),
                   static_cast<int>(mat.bandwidth),
                   &mat.data[0],
                   static_cast<int>(mat.bandwidth + 1));
    }
}
BENCHMARK(BM_Lapacke)->Iterations(100)->Apply(custom_args);

BENCHMARK_MAIN();

