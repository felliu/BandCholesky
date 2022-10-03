#include "PB_matrix.h"
#include "matrix_generator.h"
#include "par_cholesky.h"
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

#ifdef USE_PLASMA_
#include "plasma.h"
#endif

//When using PLASMA or BLIS, we use wrapper functions for LAPACK routines to not
//interfere with definitions of the LAPACK routines from  the BLAS implementation.
//Those wrapper functions are declared in the header below
#if defined USE_PLASMA_ || defined USE_BLIS

#include "lapack_wrapper.h"
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
    mkl_set_num_threads(6);
    LAPACKE_set_nancheck(0);
#endif
    for (auto _: state) {
        state.PauseTiming();
        mat = mat_cpy;
        state.ResumeTiming();
#if defined USE_BLIS || defined USE_PLASMA_
        dpbtrf_wrapper(LAPACK_COL_MAJOR, 'L',
                       static_cast<int>(mat.size),
                       static_cast<int>(mat.bandwidth),
                       &mat.data[0],
                       static_cast<int>(mat.bandwidth + 1));
#else
        LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'L',
                       static_cast<int>(mat.size),
                       static_cast<int>(mat.bandwidth),
                       &mat.data[0],
                       static_cast<int>(mat.bandwidth + 1));
#endif
    }
}
//BENCHMARK(BM_Lapacke)->Iterations(10)->Repetitions(10)->Apply(custom_args);
//BENCHMARK(BM_Lapacke)->Iterations(10)->Repetitions(10)->DenseRange(10, 100, 10);
BENCHMARK(BM_Lapacke)->Repetitions(10)->DenseRange(50, 200, 10)->UseRealTime();
//BENCHMARK(BM_Lapacke)->Repetitions(10)->DenseRange(50, 200, 10)->UseRealTime();
//BENCHMARK(BM_Lapacke)->Repetitions(10)->DenseRange(200, 2000, 100)->UseRealTime();

static void BM_par_pbtrf(benchmark::State& state) {
    PB_matrix<double> mat = get_random_pd_bandmat<double>(default_dim, state.range(0));
    PB_matrix<double> mat_cpy = mat;
#ifdef USE_MKL_
    LAPACKE_set_nancheck(0);
#endif
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
//BENCHMARK(BM_par_pbtrf)->Iterations(10)->Repetitions(10)->Apply(custom_args);
//BENCHMARK(BM_par_pbtrf)->Repetitions(10)->DenseRange(50, 200, 10)->UseRealTime();
//BENCHMARK(BM_par_pbtrf)->Repetitions(10)->DenseRange(200, 500, 100)->UseRealTime();
//BENCHMARK(BM_par_pbtrf)->Repetitions(10)->Arg(500)->UseRealTime();
//BENCHMARK(BM_par_pbtrf)->Repetitions(10)->DenseRange(200, 2000, 100)->UseRealTime();

int main(int argc, char** argv) {
#ifdef USE_BLIS
    bli_init();
#endif
#ifdef USE_PLASMA_
    plasma_init();
#endif
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
#ifdef USE_PLASMA_
    plasma_finalize();
#endif
#ifdef USE_BLIS
    bli_finalize();
#endif
    return 0;
}

