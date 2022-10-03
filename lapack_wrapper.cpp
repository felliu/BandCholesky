#include <lapacke.h>
int dpotrf_wrapper(int col, char uplo, int size, double* start, int ldab) {
    const int status = LAPACKE_dpotrf(col, uplo, size, start, ldab);
    return status;
}

int dpbtrf_wrapper(int col, char uplo, int size, int bandwidth, double* start, int ldab) {
    const int status = LAPACKE_dpbtrf(col, uplo, size, bandwidth, start, ldab);
    return status;
}
