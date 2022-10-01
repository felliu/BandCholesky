#ifndef DPOTRF_WRAPPER_H
#define DPOTRF_WRAPPER_H

//When we wish to use LAPACK functionality from one library, but BLAS from another
//often definitions will collide, since LAPACK implementations almost always also have BLAS symbols.
//Use this wrapper in those cases and compile this to a static library to link with for the LAPACK function

//Definitions from OPENBLAS
#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102
int dpotrf_wrapper(int col, char uplo, int width, double *ab, int ldab);

#endif
