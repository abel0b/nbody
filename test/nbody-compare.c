#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "assert.h"
#include "nbody.h"

#ifndef EPSILON
#define EPSILON 0.01
#endif

#define CHECK_READ(a,b) if ((a)<(b)) {\
    fprintf(stderr, "Read operation failed\n");\
}

/*
 * Compare two nbody iteration dumps
 */
int main(int argc, char * argv[]) {
    if(argc<3) {
        fprintf(stderr, "Missing arguments");
    } 

    FILE * input1;
    FILE * input2;
    input1 = fopen(argv[1], "rb");
    input2 = fopen(argv[2], "rb");
    
    float buffer1[6];
    float buffer2[6];

    enum Initializer initializer1, initializer2;
    int nb_particles1, nb_particles2;
    int iter1, iter2; 
    
    CHECK_READ(fread(&initializer1, sizeof(enum Initializer), 1, input1), 1); 
    CHECK_READ(fread(&nb_particles1, sizeof(enum Initializer), 1, input1), 1); 
    CHECK_READ(fread(&iter1, sizeof(enum Initializer), 1, input1), 1);
     
    CHECK_READ(fread(&initializer2, sizeof(enum Initializer), 1, input2), 1); 
    CHECK_READ(fread(&nb_particles2, sizeof(enum Initializer), 1, input2), 1); 
    CHECK_READ(fread(&iter2, sizeof(enum Initializer), 1, input2), 1);
    
    if ((initializer1 != initializer2) || (nb_particles1 != nb_particles2) || (iter1 != iter2)) {
        fprintf(stderr, "Incompatible metadata\n");
        return EXIT_FAILURE;
    }

    int i;
    for(i=0; i<nb_particles1; i++) {
        CHECK_READ(fread(buffer1, sizeof(float), 6, input1), 6);
        CHECK_READ(fread(buffer2, sizeof(float), 6, input2), 6);
        if((abs(buffer1[0]-buffer2[0]) > EPSILON) ||
            (abs(buffer1[1]-buffer2[1]) > EPSILON) ||
            (abs(buffer1[2]-buffer2[2]) > EPSILON) ||
            (abs(buffer1[3]-buffer2[3]) > EPSILON) ||
            (abs(buffer1[4]-buffer2[4]) > EPSILON) ||
            (abs(buffer1[5]-buffer2[5]) > EPSILON)) {
            
            #ifdef VERBOSE
            fprintf(stderr, "%f\n", abs(buffer1[0]-buffer2[0]));
            fprintf(stderr, "%f\n", abs(buffer1[1]-buffer2[1]));
            fprintf(stderr, "%f\n", abs(buffer1[2]-buffer2[2]));
            fprintf(stderr, "%f\n", abs(buffer1[3]-buffer2[3]));
            fprintf(stderr, "%f\n", abs(buffer1[4]-buffer2[4]));
            fprintf(stderr, "%f\n", abs(buffer1[5]-buffer2[5]));
            #endif

            return EXIT_FAILURE;
        }
    }

    fclose(input1);
    fclose(input2);
    return EXIT_SUCCESS;
}
