#ifndef PARDISO_TEST_H
#define PARDISO_TEST_H

#include <cassert>
#include <mkl.h>

#include "SparseMat.h"

template <typename T>
struct PB_matrix;

//lvalues for some integer constants so that we can pass pointers to them to pardiso
namespace {
    constexpr int maxfct = 1;
    constexpr int mnum = 1;
    constexpr int mtype = 2; //Symmetric, pos. def.
    constexpr int msglvl = 1;
    constexpr int solver_phase = 11;
    constexpr int factor_phase = 22;
    constexpr int nrhs = 1;
}

template <typename T>
void test_pardiso_factorize(const PB_matrix<T>& band_mat) {
    CSR_matrix<T> csr_mat = cvt_from_bandmat(band_mat);
    int pt[64] = {0};
    int iparm[64] = {0};
    pardisoinit(&pt[0], &mtype, &iparm[0]);
    iparm[26] = 1; //Check matrix for correctness
    iparm[34] = 1; //Use zero-based indexing
    std::vector<int> perm(csr_mat.rows);

    int num_eqs = csr_mat.rows;
    int error = 0;

    std::vector<T> x(csr_mat.cols);
    std::vector<T> rhs(csr_mat.rows);
    pardiso(&pt[0], &maxfct, &mnum, &mtype, &solver_phase, &num_eqs,
            &csr_mat.data[0], &csr_mat.row_ptrs[0], &csr_mat.col_inds[0], &perm[0], &nrhs, &iparm[0], &msglvl, &rhs[0], &x[0], &error);
    pardiso(&pt[0], &maxfct, &mnum, &mtype, &factor_phase, &num_eqs,
            &csr_mat.data[0], &csr_mat.row_ptrs[0], &csr_mat.col_inds[0], &perm[0], &nrhs, &iparm[0], &msglvl, &rhs[0], &x[0], &error);
    pardiso(&pt[0], &maxfct, &mnum, &mtype, &factor_phase, &num_eqs,
            &csr_mat.data[0], &csr_mat.row_ptrs[0], &csr_mat.col_inds[0], &perm[0], &nrhs, &iparm[0], &msglvl, &rhs[0], &x[0], &error);
}

#endif