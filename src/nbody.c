#include <stdio.h>
#include <math.h>
#include <stdlib.h> // drand48
#include <omp.h>
#include "nbody.h"

enum Initializer initializer = RANDOM_INITIALIZER;

#ifdef DUMP
FILE * output;
#endif 

void MoveParticles(const int nParticles, struct ParticleType* const particle, const float dt) {

  // Loop over particles that experience force
  for (int i = 0; i < nParticles; i++) { 

    // Components of the gravity force on particle i
    float Fx = 0, Fy = 0, Fz = 0; 
      
    // Loop over particles that exert force
    for (int j = 0; j < nParticles; j++) { 
      // No self interaction
      if (i != j) {
          // Avoid singularity and interaction with self
          float softening = 1e-20;

          // Newton's law of universal gravity
          float dx = particle[j].x - particle[i].x;
          float dy = particle[j].y - particle[i].y;
          float dz = particle[j].z - particle[i].z;
          float drSquared  = dx*dx + dy*dy + dz*dz + softening;
          
          #ifdef OPTIMIZE_POW
          const float drPower32  = sqrtf(drSquared * drSquared * drSquared);
          #else
          const float drPower32  = pow(drSquared, 3.0/2.0);
          #endif  
          
          // Calculate the net force
          Fx += dx / drPower32;  
          Fy += dy / drPower32;  
          Fz += dz / drPower32;
      }

    }

    // Accelerate particles in response to the gravitational force
    particle[i].vx += dt*Fx; 
    particle[i].vy += dt*Fy; 
    particle[i].vz += dt*Fz;
  }

  // Move particles according to their velocities
  // O(N) work, so using a serial loop
  for (int i = 0 ; i < nParticles; i++) { 
    particle[i].x  += particle[i].vx*dt;
    particle[i].y  += particle[i].vy*dt;
    particle[i].z  += particle[i].vz*dt;
  }
}
 
void dump_text(int iter, int nParticles, struct ParticleType* particle) {
    char filename[64];
    snprintf(filename, 64, "output_%d.txt", iter);

    FILE *f;
    f = fopen(filename, "w+");

    int i;
    for (i = 0; i < nParticles; i++)
    {
        fprintf(f, "%e %e %e %e %e %e\n",
                   particle[i].x, particle[i].y, particle[i].z,
		   particle[i].vx, particle[i].vy, particle[i].vz);
    }

    fclose(f);
}

#ifdef DUMP
void dump(int iter, int nParticles, struct ParticleType* particle) {
    int i;
    for (i = 0; i < nParticles; i++) {
        fwrite(&particle[i].x, sizeof(float), 1, output);
        fwrite(&particle[i].y, sizeof(float), 1, output);
        fwrite(&particle[i].z, sizeof(float), 1, output);
        fwrite(&particle[i].vx, sizeof(float), 1, output);
        fwrite(&particle[i].vy, sizeof(float), 1, output);
        fwrite(&particle[i].vz, sizeof(float), 1, output);
    }
}
#endif

int main(const int argc, const char** argv)
{

  // Problem size and other parameters
  const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  // Duration of test
  const int nSteps = (argc > 2)?atoi(argv[2]):10;
  // Particle propagation time step
  const float dt = 0.0005f;

  struct ParticleType* particle = malloc(nParticles*sizeof(struct ParticleType));

  // Initialize random number generator and particles
  srand48(0x2020);

  #ifdef DUMP
  // Open dump file
    char filename[64];
    snprintf(filename, 64, "data/nbody/%s.nbody", VERSION);
     output = fopen(filename, "wb");

    fwrite(&initializer, sizeof(enum Initializer), 1, output);
    fwrite(&nParticles, sizeof(int), 1, output);
    fwrite(&nSteps, sizeof(int), 1, output);
  #endif

  int i;
  for (i = 0; i < nParticles; i++)
  {
     particle[i].x =  2.0*drand48() - 1.0;
     particle[i].y =  2.0*drand48() - 1.0;
     particle[i].z =  2.0*drand48() - 1.0;
     particle[i].vx = 2.0*drand48() - 1.0;
     particle[i].vy = 2.0*drand48() - 1.0;
     particle[i].vz = 2.0*drand48() - 1.0;
  }
  
  // Perform benchmark
  printf("\nPropagating %d particles using 1 thread...\n\n", 
	 nParticles
	 );
  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration (warm-up)
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  for (int step = 1; step <= nSteps; step++) {

    const double tStart = omp_get_wtime(); // Start timing
    MoveParticles(nParticles, particle, dt);
    const double tEnd = omp_get_wtime(); // End timing

    const float HztoInts   = ((float)nParticles)*((float)(nParticles-1)) ;
    const float HztoGFLOPs = 20.0*1e-9*((float)(nParticles))*((float)(nParticles-1));

    if (step > skipSteps) { // Collect statistics
      rate  += HztoGFLOPs/(tEnd - tStart); 
      dRate += HztoGFLOPs*HztoGFLOPs/((tEnd - tStart)*(tEnd-tStart)); 
    }

    printf("%5d %10.3e %10.3e %8.1f %s\n", 
	   step, (tEnd-tStart), HztoInts/(tEnd-tStart), HztoGFLOPs/(tEnd-tStart), (step<=skipSteps?"*":""));
    fflush(stdout);

#ifdef DUMP
    dump(step, nParticles, particle);
#endif
  }
  rate/=(double)(nSteps-skipSteps); 
  dRate=sqrt(dRate/(double)(nSteps-skipSteps)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");
  free(particle);
  #ifdef DUMP
  fclose(output);
  #endif
  return EXIT_SUCCESS;
}


