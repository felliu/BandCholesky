#include "PB_matrix.h"
#include "par_cholesky.h"
#include "matrix_generator.h"

#include <cassert>
#include <iostream>
#include <mkl.h>
//#include <ittnotify.h>

namespace {
    /**
     * Reads an array from a binary file. The file is stored as |sz|data|, where sz is an unsigned 64-bit int.
     * If two template parameters are specified, the values in the output vector will be cast to type of the
     * second template argument.
     **/
    template <typename ValueType, typename OutType = ValueType>
    std::vector<OutType> read_binary_file(const std::string& path)
    {
        std::ifstream in_file(path, std::ios::in | std::ios::binary);
        size_t sz;
        in_file.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        std::vector<ValueType> data_in(sz);
        in_file.read(reinterpret_cast<char*>(data_in.data()), sizeof(ValueType) * sz);
        std::vector<OutType> out_vec;
        out_vec.reserve(sz);

        //Cast to the right output form. Not strictly necessary if U == T.
        std::transform(data_in.cbegin(), data_in.cend(), std::back_inserter(out_vec),
                    [](ValueType val){ return static_cast<OutType>(val); });

        return out_vec;
    }

    template <typename T>
    void dump_vector_to_file(const std::vector<T>& vec, const std::string& path, bool append=false)
    {
        std::ofstream outfile;
        if (append)
            outfile.open(path, std::ios::binary | std::ios::app);
        else
            outfile.open(path, std::ios::binary | std::ios::out);

        size_t sz = vec.size();
        outfile.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        outfile.write(reinterpret_cast<const char*>(vec.data()), sizeof(T) * sz);
    }

    constexpr int N_TRIES = 50;

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

        for (int i = 0; i < N_TRIES; ++i) {
            const double start = dsecnd();
            status = par_dpbtrf(static_cast<int>(mat.size),
                                static_cast<int>(mat.bandwidth),
                                &mat.data[0],
                                static_cast<int>(mat.bandwidth + 1));
            const double end = dsecnd();
            total_time += (end - start);
            mat = mat_cpy;
        }

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

        for (int i = 0; i < N_TRIES; ++i) {
            const double start = dsecnd();
            status = LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'L',
                                    static_cast<int>(mat.size),
                                    static_cast<int>(mat.bandwidth),
                                    &mat.data[0],
                                    static_cast<int>(mat.bandwidth + 1));
            const double end = dsecnd();
            total_time += (end - start);
            mat = mat_cpy;
        }

        return total_time;
    }

    template <typename T>
    void test_factorization(PB_matrix<T>& mat) {
        std::cout << "Testing factorization correctness...\n";
        int status = 0;
        auto mat_cpy = mat;
        status = par_dpbtrf(static_cast<int>(mat.size),
                            static_cast<int>(mat.bandwidth),
                            &mat.data[0],
                            static_cast<int>(mat.bandwidth + 1));
        assert(status == 0);
        status = LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'L',
                                static_cast<int>(mat_cpy.size),
                                static_cast<int>(mat_cpy.bandwidth),
                                &mat_cpy.data[0],
                                static_cast<int>(mat_cpy.bandwidth + 1));
        assert(status == 0);

        double diff = 0.0;
        for (int i = 0; i < mat.data.size(); ++i)
            diff += std::abs(mat.data[i] - mat_cpy.data[i]);

        std::cout << "Total diff: " << diff << "\n";
        std::cout << "Avg diff / elem: " << diff / static_cast<double>(mat.data.size()) << "\n";

        mat = mat_cpy;
    }
}

int main(int argc, char* argv[]) {
    if (argc > 3 || argc == 1) {
        std::cerr << "Usage: ./band_cholesky_test <filename>\n"
                     "    \t ./band_cholesky_test <size> <bandwidth>\n";
        return -1;
    }

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
    //TODO:fix
    double total_time_ours = run_parallel_benchmark(N_TRIES, mat);
    double total_time_MKL = run_MKL_benchmark(N_TRIES, mat);

    std::cout << "Elapsed time (ours): " <<  total_time_ours << " s, Avg / iter: " << 1000.0 * total_time_ours / static_cast<double>(N_TRIES) << " ms\n";
    std::cout << "Elapsed time (MKL): " <<  total_time_MKL << " s, Avg / iter: " << 1000.0 * total_time_MKL / static_cast<double>(N_TRIES) << " ms\n";

    //test_factorization(mat);

    return 0;
}
