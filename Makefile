all: nbody nbody-openacc nbody-cuda nbody-cuda-structofarray

nbody: nbody.c
	gcc nbody.c -o nbody -lm -fopenmp -O3

nbody-openacc: nbody-openacc.c
	pgcc -acc nbody-openacc.c -o nbody-openacc -lm -O3

nbody-cuda: nbody-cuda.cu
ifdef dump
	nvcc -DDUMP nbody-cuda.cu -o nbody-cuda -lm
else
	nvcc nbody-cuda.cu -o nbody-cuda -lm
endif

nbody-cuda-structofarray: nbody-cuda-structofarray.cu
ifdef dump
	nvcc -DDUMP $< -o $@ -lm
else
	nvcc $< -o $@ -lm
endif

clean:
	rm -f nbody nbody-openacc nbody-cuda nbody-cuda-structofarray *.txt

