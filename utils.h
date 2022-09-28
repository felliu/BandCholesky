#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "PB_matrix.h"

#ifdef USE_MKL_
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
/**
 * Reads an array from a binary file. The file is stored as |sz|data|,
 * where sz is an unsigned 64-bit int.
 * If two template parameters are specified,
 * the values in the output vector will be cast to type of the second template argument.
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
/** 
 * Dumps an array to a binary file stored as |sz|data|, where
 * sz is an unsigned 64-bit integer.
 **/

template <typename T>
void dump_vector_to_file(const std::vector<T>& vec, const std::string& path,
                         bool append=false)
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

template <typename T>
void test_factorization(const PB_matrix<T>& mat, PB_matrix<T>& mat_orig) {
    std::cout << "Testing factorization correctness...\n";
    int status =
        LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'L',
                       static_cast<int>(mat_orig.size),
                       static_cast<int>(mat_orig.bandwidth),
                       &mat_orig.data[0],
                       static_cast<int>(mat_orig.bandwidth + 1));
    assert(status == 0);

    double diff = 0.0;
    for (size_t i = 0; i < mat.data.size(); ++i)
        diff += std::abs(mat.data[i] - mat_orig.data[i]);

    std::cout << "Total diff: " << diff << "\n";
    std::cout << "Avg diff / elem: " << diff / static_cast<double>(mat.data.size()) << "\n";
}

#endif
