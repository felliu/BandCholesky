CXX = g++
CXX_FLAGS = -O3 -g -Wall -fopenmp -m64 -I"${MKLROOT}/include"

LD_FLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -liomp5
PAR_LD_FLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

band_cholesky_test: main.o par_cholesky.o Makefile
	${CXX} ${LD_FLAGS} main.o par_cholesky.o -o band_cholesky_test

main.o: main.cpp PB_matrix.h par_cholesky.h matrix_generator.h Makefile
	${CXX} ${CXX_FLAGS} main.cpp -c -o main.o

par_cholesky.o: par_cholesky.cpp par_cholesky.h PB_matrix.h Makefile
	${CXX} ${CXX_FLAGS} par_cholesky.cpp -c -o par_cholesky.o