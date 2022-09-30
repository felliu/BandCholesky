cmake_minimum_required(VERSION 3.21)

project(BandCholesky LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

OPTION(BC_USE_MKL "Use MKL for BLAS and LAPACK" FALSE)
OPTION(BC_USE_BLIS "Use specific BLIS kernel internally" FALSE)

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

#We want to be able to run with both GNU OpenMP and Intel OpenMP, handle that here
if(BC_USE_GOMP)
    #GNU OpenMP which should be what FindOpenMP gets ...
    find_package(OpenMP REQUIRED)
    message(STATUS "Expecting GNU OpenMP: ${OpenMP_CXX_LIB_NAMES}")
    target_link_libraries(band_cholesky_test PRIVATE OpenMP::OpenMP_CXX)
else()
    #Assume we want Intel OpenMP
    find_library(IntelOpenMP NAMES iomp5
                 HINTS ENV LIBRARY_PATH)
    message(STATUS "Intel OpenMP: ${IntelOpenMP}")
    target_link_libraries(band_cholesky_test PRIVATE ${IntelOpenMP})
endif()
if (BC_USE_MKL)
    target_compile_definitions(band_cholesky_test PRIVATE USE_MKL_)
endif()
if (BC_USE_BLIS)
    find_library(BLIS NAMES blis
                 HINTS ENV LIBRARY_PATH)
    message(STATUS "BLIS: ${BLIS}")
    target_link_libraries(band_cholesky_test PRIVATE ${BLIS})
    target_compile_definitions(band_cholesky_test PRIVATE USE_BLIS)
endif()

target_link_libraries(band_cholesky_test PRIVATE BLAS::BLAS benchmark::benchmark)
target_compile_options(band_cholesky_test PRIVATE -fopenmp)
