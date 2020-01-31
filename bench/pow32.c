#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define N 1000000


// Compare 3/2 power calculation

int main() {
    float start, end;
    int i;
    int a = 42;
    float ret;

    start = omp_get_wtime();
    for(i=0;i<N;i++) {
        ret = pow(a, 3.0/2.0);
    }
    end = omp_get_wtime();
    printf("POW: %f in %fs\n", ret, end-start);

    start = omp_get_wtime();
    for(i=0;i<N;i++) {
        ret = sqrtf(a*a*a);
    }
    end = omp_get_wtime();
    printf("SQRT: %f in %fs\n", ret, end-start);

    return 0;
}
