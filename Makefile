cflags=-Iinclude -lm
ifndef nodump
	cflags+=-DDUMP
endif

.PHONY: all
all: data/nbody bin/nbody-baseline bin/nbody-baseline-optimized bin/nbody-openacc bin/nbody-cuda bin/nbody-cuda-soa bin/nbody-cuda-soa-optimized bin/nbody-compare

data/nbody:
	mkdir -p data/nbody

bin/nbody-baseline: src/nbody.c
	gcc $(cflags) -O3 -march=native -fopenmp -DVERSION='"baseline"' $^ -o $@

bin/nbody-baseline-optimized: src/nbody.c
	gcc $(cflags) -O3 -march=native -fopenmp -DOPTIMIZE_POW -DVERSION='"baseline-optimized"' $^ -o $@

bin/nbody-openacc: src/nbody-openacc.c
	pgcc $(cflags) -O3 -DOPTIMIZE_POW -DVERSION='"openacc"' -acc $^ -o $@

bin/nbody-cuda: src/nbody-cuda.cu
	nvcc $(cflags) -O3 -DOPTIMIZE_POW -DVERSION='"cuda"' $^ -o $@

bin/nbody-cuda-soa: src/nbody-cuda-soa.cu
	nvcc $(cflags) -O3 -DOPTIMIZE_POW -DVERSION='"cuda-soa"' $^ -o $@

bin/nbody-cuda-soa-optimized: src/nbody-cuda-soa-optimized.cu
	nvcc $(cflags) -O3 -DOPTIMIZE_POW -DVERSION='"cuda-soa-optimized"' $^ -o $@

bin/nbody-compare: test/nbody-compare.c
	gcc $(cflags) $^ -o $@

.PHONY: test
test: all
	./test/test.sh

.PHONY: bench
bench: all
	./bench/bench.sh

.PHONY: clean
clean:
	rm -f bin/* data/*.nbody *.txt

