cmake_minimum_required(VERSION 3.21)

project(BandCholesky LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

OPTION(BC_USE_PLASMA "Link with PLASMA for LAPACK" FALSE)
OPTION(BC_USE_IOMP "Use Intel OpenMP" FALSE)

find_package(BLAS REQUIRED)
message(STATUS BLAS lib: ${BLAS_LIBRARIES})


if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR
   "${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model=precise")
endif()

set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(external/benchmark)

add_library(par_band_cholesky SHARED
    par_cholesky.cpp
    par_cholesky.h)

add_executable(band_cholesky_test
    gbench_main.cpp
    matrix_generator.h
    PB_matrix.h
    run_benchmarks.h
    utils.h)

target_link_libraries(band_cholesky_test PRIVATE BLAS::BLAS)
target_link_libraries(par_band_cholesky PRIVATE BLAS::BLAS)

#Identify the BLAS vendor requested to set preprocessor defines
if("${BLA_VENDOR}" MATCHES "Intel10.*")
    target_compile_definitions(band_cholesky_test PRIVATE USE_MKL_)
    target_compile_definitions(par_band_cholesky PRIVATE USE_MKL_)
elseif ("${BLA_VENDOR}" STREQUAL "FLAME")
    #This is the name used for BLIS
    target_compile_definitions(par_band_cholesky PRIVATE USE_BLIS_)
    target_compile_definitions(band_cholesky_test PRIVATE USE_BLIS_)

    #BLIS does not provide LAPACK so link in separately from OpenBLAS
    set(BLA_VENDOR OpenBLAS)
    set(BLA_STATIC ON)
    #For some reason, find_package LAPACK doesn't work with BLA_STATIC and
    #OpenBLAS
    find_package(BLAS REQUIRED)
    message(STATUS "${BLAS_LIBRARIES}")

    #This wrapper is a way to avoid collisions between BLAS symbols in OpenBLAS and BLIS
    #We link a separate static library with OpenBLAS for the LAPACK functions we need,
    #and provide new names for them to link back to the original libraries..
    add_library(lapack_wrapper STATIC lapack_wrapper.cpp lapack_wrapper.h)
    target_link_libraries(lapack_wrapper PRIVATE "${BLAS_LIBRARIES}")
    target_link_libraries(par_band_cholesky PRIVATE lapack_wrapper)
    target_link_libraries(band_cholesky_test PRIVATE lapack_wrapper)
endif()


#We want to be able to run with both GNU OpenMP and Intel OpenMP, handle that here
if (BC_USE_IOMP)
    find_library(IntelOpenMP NAMES iomp5
             HINTS ENV LIBRARY_PATH)
    target_link_libraries(par_band_cholesky PRIVATE ${IntelOpenMP})
    message(STATUS "Expecting IntelOpenMP: ${IntelOpenMP}")
else()
    #Otherwise fall back to some whatever other OpenMP available on the system
    #Often probably GNU OpenMP
    find_package(OpenMP REQUIRED)
    message(STATUS "Expecting GNU OpenMP: ${OpenMP_CXX_LIB_NAMES}")
    target_link_libraries(band_cholesky_test PRIVATE OpenMP::OpenMP_CXX)
endif()

if (BC_USE_PLASMA)
    find_library(PLASMA NAMES plasma
        HINTS ENV LIBRARY_PATH REQUIRED)
    target_compile_definitions(band_cholesky_test PRIVATE USE_PLASMA_)

    target_link_libraries(band_cholesky_test PRIVATE ${PLASMA})
endif()

target_link_libraries(band_cholesky_test PRIVATE par_band_cholesky)
target_link_libraries(band_cholesky_test PRIVATE benchmark::benchmark)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(band_cholesky_test PRIVATE -fopenmp)
    target_compile_options(par_band_cholesky PRIVATE -fopenmp)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR
       "${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
    target_compile_options(band_cholesky_test PRIVATE -qopenmp)
    target_compile_options(par_band_cholesky PRIVATE -qopenmp)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(band_cholesky_test PRIVATE -fopenmp)
    target_compile_options(par_band_cholesky PRIVATE -fopenmp)
endif()
