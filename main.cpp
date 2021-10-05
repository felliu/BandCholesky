#include "PB_matrix.h"
#include "par_cholesky.h"

#include <cassert>
#include <iostream>
#include <mkl.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./band_cholesky_test <filename>\n";
        return -1;
    }

    std::string filename(argv[1]);
    PB_matrix<double> mat = read_pb_matrix<double>(filename);
    pad_pb_matrix(mat);
    PB_matrix<double> mat_cpy(mat);
    std::cerr << "Rows: " << mat.bandwidth + 1 << "\n";
    std::cerr << "Columns: " << mat.size << "\n";
    std::cerr << "Data size: " << mat.data.size() << "\n";
    int status = par_dpbtrf(static_cast<int>(mat.size),
                            static_cast<int>(mat.bandwidth), &mat.data[0],
                            static_cast<int>(mat.bandwidth + 1));
    std::cerr << "Status: " << status << "\n";
    assert(status == 0);
    status = LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'L', static_cast<int>(mat_cpy.size),
                            static_cast<int>(mat_cpy.bandwidth), &mat_cpy.data[0],
                            static_cast<int>(mat_cpy.bandwidth + 1));
    assert(status == 0);

    std::cerr << mat.data[0] << "\n";
    std::vector<double> rhs(mat.size, 1.0);
    std::vector<double> rhs2(mat_cpy.size, 0.0);
    status = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'L', 'N', 'N', static_cast<int>(mat.size), static_cast<int>(mat.bandwidth), 1,
                            &mat.data[0], static_cast<int>(mat.bandwidth +1), &rhs[0], rhs.size());
    assert(status == 0);

    status = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'L', 'N', 'N', static_cast<int>(mat_cpy.size), static_cast<int>(mat_cpy.bandwidth), 1,
                            &mat_cpy.data[0], static_cast<int>(mat_cpy.bandwidth + 1), &rhs2[0], rhs2.size());
    assert(status == 0);

    double diff = 0.0;
    for (int i = 0; i < rhs.size(); ++i)
        diff += std::abs(rhs[i] - rhs2[i]);

    std::cerr << "Diff: " << diff << "\n";
    std::cerr << "Avg diff / elem: " << diff / static_cast<double>(rhs.size()) << "\n";
    return 0;
}