#include "PB_matrix.h"
#include "par_cholesky.h"
#include "matrix_generator.h"

#include <cassert>
#include <iostream>
#include <mkl.h>

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

    constexpr int N_TRIES = 1000;
}

int main(int argc, char* argv[]) {
    if (argc > 3 || argc == 1) {
        std::cerr << "Usage: ./band_cholesky_test <filename>\n"
                     "    \t ./band_cholesky_test <size> <bandwidth>\n";
        return -1;
    }
    mkl_set_num_threads(1);

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

    double start = dsecnd();
    int status = 0;
    double total_time = 0.0;
    for (int i = 0; i < N_TRIES; ++i) {
        double start = dsecnd();
#ifdef RUN_PARALLEL
        status = par_dpbtrf(static_cast<int>(mat.size),
                            static_cast<int>(mat.bandwidth), &mat.data[0],
                            static_cast<int>(mat.bandwidth + 1));
#else
        status = LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'L',
                                static_cast<int>(mat.size),
                                static_cast<int>(mat.bandwidth),
                                &mat.data[0],
                                static_cast<int>(mat.bandwidth + 1));
#endif
        assert(status == 0);
        double end = dsecnd();
        total_time += (end - start);
        mat = mat_cpy;
    }
    std::cout << "Elapsed time: " <<  total_time << " s\n";

    std::vector<double> rhs(mat.size, 100.0);
    std::vector<double> rhs2(mat_cpy.size, 100.0);
    status = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'L', 'N', 'N', static_cast<int>(mat.size), static_cast<int>(mat.bandwidth), 1,
                            &mat.data[0], static_cast<int>(mat.bandwidth +1), &rhs[0], rhs.size());
    assert(status == 0);

    double diff = 0.0;
    for (size_t i = 0; i < rhs.size(); ++i)
        diff += std::abs(rhs[i] - rhs2[i]);

    std::cerr << "Diff: " << diff << "\n";
    std::cerr << "Avg diff / elem: " << diff / static_cast<double>(rhs.size()) << "\n";
    return 0;
}
