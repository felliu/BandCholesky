#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <algorithm>
#include <vector>
#include <utility>

template <typename T>
struct PB_matrix;

template <typename T>
struct CSR_matrix {
    size_t rows;
    size_t cols;

    std::vector<double> data;
    std::vector<int> col_inds;
    std::vector<int> row_ptrs;
};

template <typename T>
CSR_matrix<T> cvt_from_bandmat(const PB_matrix<T>& band_mat) {
    static constexpr T tolerance = 0.0;

    CSR_matrix<T> mat;
    mat.rows = band_mat.size;
    mat.cols = band_mat.size;
    int old_row = -1;
    int nz_count = 0;
    for (int i = 0; i < band_mat.data.size(); ++i) {
        auto [row, col] = band_mat.get_row_col(i);
        //We store the upper half of the matrix, so transpose if not in the upper half
        if (row > col)
            std::swap(row, col);

        if (row != old_row)
            mat.row_ptrs.push_back(nz_count);

        if (std::abs(band_mat.data[i]) > tolerance) {
            mat.data.push_back(band_mat.data[i]);
            mat.col_inds.push_back(col);
            ++nz_count;
        }

        old_row = row;
    }

    //Final entry of row_ptr should be the number of nonzeros
    mat.row_ptrs.push_back(nz_count);
    std::cerr << "data sz: " << mat.data.size() << "\n";
    std::cerr << "col inds sz: " << mat.col_inds.size() << "\n";
    std::cerr << "row ptrs sz: " << mat.row_ptrs.size() << "\n";

    std::cerr << "Max col ind: " << *std::max_element(mat.col_inds.cbegin(), mat.col_inds.cend()) << "\n";
    std::cerr << "Max row ptr: " << *std::max_element(mat.row_ptrs.cbegin(), mat.row_ptrs.cend()) << "\n";
    return mat;
}


#endif