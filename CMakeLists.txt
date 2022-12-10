cmake_minimum_required(VERSION 3.21)

project(BandCholesky LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

OPTION(BC_USE_MKL "Use MKL for BLAS and LAPACK" FALSE)
OPTION(BC_USE_BLIS "Use specific BLIS kernel internally" FALSE)
OPTION(BC_USE_PLASMA "Link with PLASMA for LAPACK" FALSE)

#For now we only support using PLASMA for LAPACK when BLIS is used for BLAS
#to avoid a mess with LAPACK symbols from different libraries interfering at runtime.

if (BC_USE_BLIS)
    set(BLA_STATIC ON)
endif()
find_package(BLAS REQUIRED)
message(STATUS BLAS lib: ${BLAS_LIBRARIES})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR
   "${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model=precise")
endif()

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
#We treat BLIS as a special case for blas implementations,
#since then we need DPOTRF from somewhere else, and we use a wrapper to avoid symbol redefinitions
if (BC_USE_BLIS)
    find_library(BLIS NAMES blis
                 HINTS ENV LIBRARY_PATH)
    message(STATUS "BLIS: ${BLIS}")
    target_link_libraries(band_cholesky_test PRIVATE ${BLIS})
    target_compile_definitions(band_cholesky_test PRIVATE USE_BLIS)

    add_library(lapack_wrapper STATIC lapack_wrapper.cpp lapack_wrapper.h)
    #Link in the external LAPACK to the wrapper. Use private to not link the main target to LAPACK too.
    target_link_libraries(lapack_wrapper PRIVATE BLAS::BLAS)

    target_link_libraries(band_cholesky_test PRIVATE lapack_wrapper)
else()
    target_link_libraries(band_cholesky_test PRIVATE BLAS::BLAS)
endif()

if (BC_USE_PLASMA)
    find_library(PLASMA NAMES plasma
                 HINTS ENV LIBRARY_PATH)
    message(STATUS "Plasma: ${PLASMA}")
    target_compile_definitions(band_cholesky_test PRIVATE USE_PLASMA_)
    target_link_libraries(band_cholesky_test PRIVATE ${PLASMA})
endif()

target_link_libraries(band_cholesky_test PRIVATE benchmark::benchmark)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(band_cholesky_test PRIVATE -fopenmp)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR
       "${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
    target_compile_options(band_cholesky_test PRIVATE -qopenmp)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(band_cholesky_test PRIVATE -fopenmp)
endif()
