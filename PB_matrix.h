#ifndef PB_MATRIX_H
#define PB_MATRIX_H

#include <cassert>
#include <fstream>
#include <string>
#include <cstdint>
#include <algorithm>
#include <vector>

/*  Positive-definite symmetric band matrix.
    Only stores lower triangular elements in column major order.
*/
template <typename ValueType>
struct PB_matrix
{
    size_t size;
    int32_t bandwidth;
    std::vector<ValueType> data;

    void set_element(int row, int col, ValueType val) {
        data[get_flat_index(row, col)] = val;
    }

    int get_flat_index(int row, int col) const;
};

template <typename ValueType>
int PB_matrix<ValueType>::get_flat_index(int row, int col) const {
    assert((row < this->size) && (col < this->size));
    if (row < col)
        std::swap(row, col);

    assert(row - col <= this->bandwidth);
    const int bandoffset = row - col;
    return col * (this->bandwidth + 1) + bandoffset;
}

template <typename ValueType>
PB_matrix<ValueType> read_pb_matrix(const std::string& path)
{
    PB_matrix<ValueType> matrix;
    std::ifstream infile(path, std::ios::binary | std::ios::in);
    infile.read(reinterpret_cast<char*>(&matrix.size), sizeof(uint64_t));
    infile.read(reinterpret_cast<char*>(&matrix.bandwidth), sizeof(int32_t));

    uint64_t num_elems;
    infile.read(reinterpret_cast<char*>(&num_elems), sizeof(uint64_t));
    matrix.data.resize(num_elems);
    infile.read(reinterpret_cast<char*>(&matrix.data[0]), sizeof(double) * num_elems);
    return matrix;
}

/*
    Pads the flat data array with the symmetric band matrix elements to size (bandwidth + 1) x num_rows
    This is done by storing each band of the matrix in its own column, and padding with zeroes for columns where the bands are less
    than bandwidth + 1 long.

    Parameters:
    matrix - the matrix to pad (input/output parameter)
*/
template <typename ValueType>
void pad_pb_matrix(PB_matrix<ValueType>& matrix)
{
    std::vector<ValueType> padded_data(matrix.size * (matrix.bandwidth + 1), 0.0);

    //For the first (num_rows - bandwidth) rows, no padding is necessary, so we can just copy over the data
    int n_cpy_elems = (matrix.size - matrix.bandwidth) * (matrix.bandwidth + 1);
    std::copy(matrix.data.cbegin(), matrix.data.cbegin() + n_cpy_elems, padded_data.begin());

    //Keep track of the current index in the original array that we are writing to the padded array.
    int unpadded_idx = n_cpy_elems;
    int num_zeros = 1;
    for (int row = matrix.size - matrix.bandwidth; row < matrix.size; ++row)
    {
        //Start index in the flat array for each row
        int start_idx = row * (matrix.bandwidth + 1);

        //The number of elements
        int end_idx = start_idx + (matrix.bandwidth + 1) - num_zeros;
        for (int i = start_idx; i < end_idx; ++i) {
            padded_data[i] = matrix.data[unpadded_idx];
            ++unpadded_idx;
        }
        //Each new row in the padded array has one more zero than the previous.
        ++num_zeros;
    }

    matrix.data = std::move(padded_data);
}

#endif