#include "PB_matrix.h"
#include "par_cholesky.h"
#include "matrix_generator.h"
#include "run_benchmarks.h"
#include "utils.h"

#include <cassert>
#include <iostream>
#include <mkl.h>
#ifdef USE_PLASMA
#include <plasma.h>
#endif

int main(int argc, char* argv[]) {
    constexpr int N_TRIES = 5;
    if (argc > 3 || argc == 1) {
        std::cerr << "Usage: ./band_cholesky_test <filename>\n"
                     "    \t ./band_cholesky_test <size> <bandwidth>\n";
        return -1;
    }
#ifdef USE_PLASMA
    plasma_init();
#endif

    LAPACKE_set_nancheck(0);
    PB_matrix<double> mat, mat_cpy;

    if (argc == 2) {
        std::string filename(argv[1]);
        mat = read_pb_matrix<double>(filename);
        pad_pb_matrix(mat);
    } else if (argc == 3) {
        auto tmp = std::atoll(argv[1]);
        int bandwidth = std::atoi(argv[2]);
        if (tmp < 0 || bandwidth < 0) {
            std::cerr << "Provide positive values for size and bandwidth\n.";
            return -1;
        }

        size_t dimension = static_cast<size_t>(tmp);
        mat = get_random_pd_bandmat<double>(dimension, bandwidth);
    }

    mat_cpy = mat;
    //double total_time_ours = run_parallel_benchmark(N_TRIES, mat);
    double total_time_MKL = run_MKL_benchmark(N_TRIES, mat);
    //double total_time_plasma = run_plasma_benchmark(N_TRIES, mat);

    //std::cout << "Elapsed time (ours): " <<  total_time_ours << " s, Avg / iter: " << 1000.0 * total_time_ours / static_cast<double>(N_TRIES) << " ms\n";
    //std::cout << "Elapsed time (MKL): " <<  total_time_MKL << " s, Avg / iter: " << 1000.0 * total_time_MKL / static_cast<double>(N_TRIES) << " ms\n";
    //std::cout << "Elapsed time (plasma): " <<  total_time_plasma << " s, Avg / iter: " << 1000.0 * total_time_plasma / static_cast<double>(N_TRIES) << " ms\n";
    test_factorization(mat, mat_cpy);

#ifdef USE_PLASMA
    plasma_finalize();
#endif
    return 0;
}
