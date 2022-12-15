# Cholesky Factorization of Banded Matrices
An implementation of Cholesky factorization of banded matrices using OpenMP tasks for parallelism.

## Requirements
A working BLAS installation is needed, as well as the LAPACK routine dpotrf. The CMake build system tries to find these based on the users preference (more details in the build section).

## Build
Building is done in a traditional CMake fashion

`mkdir build && cd build`

`cmake <options> ..`

`cmake --build .`

It is recommended to specify a BLAS implementation to try to link against using `-DBLA_VENDOR=<...>`, where different options for `<...>` can be found in the [CMake documentation](https://cmake.org/cmake/help/latest/module/FindBLAS.html).

So for example to build using Intel MKL as a backend:

`mkdir build && cd build`

`cmake -DBLA_VENDOR=Intel10_64lp_seq ..`

`cmake --build .`

The BLAS backends need to be installed separately on the system before building this library!

## Benchmarks
Code for benchmarking is provided using the [Google Benchmark](https://github.com/google/benchmark) framework. The benchmark is built with the same CMake build system, and the benchmarking code is mainly in gbench_main.cpp.
