#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include "PB_matrix.h"
#include <random>
#include <iostream>

template <typename ValueType>
PB_matrix<ValueType> get_random_pd_bandmat(size_t dimension, int bandwidth) {
    PB_matrix<ValueType> mat;
    mat.size = dimension;
    mat.bandwidth = bandwidth;
    mat.data = std::vector<ValueType>(dimension * (bandwidth + 1));

    //Set up the RNG
    std::random_device rng;
    std::mt19937 generator(rng());
    std::uniform_real_distribution<ValueType> distr(0.0, 5.0);

    for (size_t i = 0; i < dimension; ++i) {
        for (size_t j = i; j < std::min(static_cast<size_t>(i + bandwidth), dimension); ++j) {
            //Set a very large value on the diagonal which is large enough to be larger than the sum
            //of the off-diagonal elements on the same row. This guarantees pos. definiteness.
            if (i == j)
                mat.set_element(i, j, static_cast<double>(bandwidth) * 10.0);
            else
                mat.set_element(i, j, distr(generator));
        }
    }

    return mat;
}

#endif
