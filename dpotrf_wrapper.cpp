#include <lapacke.h>

int dpotrf_wrapper(int col, char uplo, int width, double* start, int ldab) {
    const int status = LAPACKE_dpotrf(col, uplo, width, start, ldab);
    return status;
}
