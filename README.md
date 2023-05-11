# Cholesky Factorization of Banded Matrices
An implementation of Cholesky factorization of banded matrices using OpenMP tasks for parallelism.

## Requirements
A working BLAS installation is needed, as well as the LAPACK routine `dpotrf`. The CMake build system tries to find these based on the users preference (more details in the build section).
The most complicated backend for this that we've found is BLIS, since it doesn't provide `dpotrf` natively.

## Build
Building is done in a traditional CMake fashion

`mkdir build && cd build`

`cmake <options> ..`

`cmake --build .`

It is recommended to specify a BLAS implementation to try to link against using `-DBLA_VENDOR=<...>`, where different options for `<...>` can be found in the [CMake documentation](https://cmake.org/cmake/help/latest/module/FindBLAS.html).

For example to build using Intel MKL as a backend:

`mkdir build && cd build`

`cmake -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release -DBC_USE_IOMP ..`

`cmake --build .`

The BLAS backends need to be installed separately on the system before building this library!

If BLIS is used as a backend, the CMakeLists is hardcoded to try to link to OpenBLAS for `dpotrf` as a workaround. So ensure that both of those are installed if using BLIS.

## Benchmarks
Code for benchmarking is provided using the [Google Benchmark](https://github.com/google/benchmark) framework. The benchmark is built with the same CMake build system, and the benchmarking code is mainly in gbench_main.cpp. Many of the available flags are specified in the [user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md) for Google Benchmark.

When benchmarking, we recommend building with the `-DCMAKE_BUILD_TYPE=Release` flag. Our experience suggests the best performance (on Intel CPUs) when using sequential Intel MKL as a backend with Intel OpenMP.
CMake will try to build with Intel OpenMP when the flag `-DBC_USE_IOMP=TRUE` is set. On AMD, other backends may perform better (we've seen encouraging results with OpenBLAS and clang (`cmake -DBLA_VENDOR=OpenBLAS -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release ..`), for instance).
