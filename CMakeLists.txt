cmake_minimum_required(VERSION 3.21)

project(BandCholesky LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(BLAS REQUIRED)
message(STATUS BLAS lib: ${BLAS_LIBRARIES})

set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(external/benchmark)

add_executable(band_cholesky_test
    gbench_main.cpp
    par_cholesky.cpp
    par_cholesky.h
    matrix_generator.h
    PB_matrix.h
    run_benchmarks.h
    utils.h)

target_link_libraries(band_cholesky_test PRIVATE BLAS::BLAS benchmark::benchmark)

