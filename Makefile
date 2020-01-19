cflags = -Iinclude -lm
ifndef nodump
	cflags += -DDUMP
endif

.PHONY: all
all: bin/nbody-baseline bin/nbody-openacc bin/nbody-cuda bin/nbody-cuda-soa bin/nbody-compare

bin/nbody-baseline: src/nbody.c
	gcc $(cflags) -DVERSION='"baseline"' $^ -o $@ -fopenmp -O3

bin/nbody-openacc: src/nbody-openacc.c
	pgcc $(cflags) -DVERSION='"openacc"' -acc $^ -o $@ -O3

bin/nbody-cuda: src/nbody-cuda.cu
	nvcc $(cflags) -DVERSION='"cuda"' $^ -o $@

bin/nbody-cuda-soa: src/nbody-cuda-soa.cu
	nvcc $(cflags) -DVERSION='"cuda-soa"' $^ -o $@

bin/nbody-compare: test/nbody-compare.c
	gcc $(cflags) $^ -o $@

.PHONY: test
test: all
	./test/test.sh

.PHONY: clean
clean:
	rm -f bin/* data/*.nbody *.txt

