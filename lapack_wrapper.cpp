#ifdef USE_PLASMA_
#include <plasma.h>
#else
#include <lapacke.h>
#endif

int dpotrf_wrapper(int col, char uplo, int size, double* start, int ldab) {
#ifdef USE_PLASMA_
    plasma_enum_t uplo_plasma = plasma_uplo_const(uplo);
    const int status = plasma_dpotrf(uplo_plasma, size, start, ldab);
#else
    const int status = LAPACKE_dpotrf(col, uplo, size, start, ldab);
#endif
    return status;
}

int dpbtrf_wrapper(int col, char uplo, int size, int bandwidth, double* start, int ldab) {
#ifdef USE_PLASMA_
    plasma_enum_t uplo_plasma = plasma_uplo_const(uplo);
    const int status = plasma_dpbtrf(uplo_plasma, size, bandwidth, start, ldab);
#else
    const int status = LAPACKE_dpbtrf(col, uplo, size, bandwidth, start, ldab);
#endif
    return status;
}
