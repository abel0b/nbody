#ifndef NBODY_H
#define NBODY_H

#ifndef VERSION
#define VERSION "undefined"
#endif

enum Initializer {
    RANDOM_INITIALIZER,
};

struct ParticleType { 
    float x, y, z;
    float vx, vy, vz; 
};


struct ParticleArray {
    float * x;
    float * y;
    float * z;
    float * vx;
    float * vy;
    float * vz; 
};

void dump_text(int iter, int nParticles, struct ParticleType* particle);

void dump(int iter, int nParticles, struct ParticleType* particle);

 #endif
