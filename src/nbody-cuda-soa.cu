#include <stdio.h>
#include <math.h>
#include <stdlib.h> // drand48
#include <omp.h>
#include <sys/time.h>
#include "nbody.h"

enum Initializer initializer = RANDOM_INITIALIZER;

__global__ void UpdateParticle(const int nParticles, struct ParticleArray * const particle, const float dt) {
    int i; 
    int stride = blockDim.x * gridDim.x;

    // Loop over particles that experience force
    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nParticles; i += stride) {
        // Components of the gravity force on particle i
    	float Fx = 0, Fy = 0, Fz = 0; 
      
    	// Loop over particles that exert force
    	for (int j = 0; j < nParticles; j++) { 
      	    // No self interaction
      	    if (i != j) {
		        // Avoid singularity and interaction with self
		        const float softening = 1e-20;

                // Newton's law of universal gravity
                const float dx = particle->x[j] - particle->x[i];
                const float dy = particle->y[j] - particle->y[i];
                const float dz = particle->z[j] - particle->z[i];
                const float drSquared  = dx*dx + dy*dy + dz*dz + softening;
                const float drPower32  = pow(drSquared, 3.0/2.0);
                
                // Calculate the net force
                Fx += dx / drPower32;  
                Fy += dy / drPower32;  
                Fz += dz / drPower32;
            }
        }

        // Accelerate particles in response to the gravitational force
        particle->vx[i] += dt*Fx; 
        particle->vy[i] += dt*Fy; 
        particle->vz[i] += dt*Fz;
    }
}

void MoveParticles(const int nParticles, struct ParticleArray * const particle, const float dt) {
    struct ParticleArray gpu_particle_tmp;
    cudaMalloc(&gpu_particle_tmp.x, sizeof(float) * nParticles);
    cudaMalloc(&gpu_particle_tmp.y, sizeof(float) * nParticles);
    cudaMalloc(&gpu_particle_tmp.z, sizeof(float) * nParticles);
    cudaMalloc(&gpu_particle_tmp.vx, sizeof(float) * nParticles);
    cudaMalloc(&gpu_particle_tmp.vy, sizeof(float) * nParticles);
    cudaMalloc(&gpu_particle_tmp.vz, sizeof(float) * nParticles);
    
    struct ParticleArray * gpu_particle;
    cudaMalloc(&gpu_particle, sizeof(struct ParticleArray));
    cudaMemcpy(gpu_particle, &gpu_particle_tmp, sizeof(struct ParticleArray), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_particle_tmp.x, particle->x, sizeof(float) * nParticles, cudaMemcpyHostToDevice); 
    cudaMemcpy(gpu_particle_tmp.y, particle->y, sizeof(float) * nParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_particle_tmp.z, particle->z, sizeof(float) * nParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_particle_tmp.vx, particle->vx, sizeof(float) * nParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_particle_tmp.vy, particle->vy, sizeof(float) * nParticles, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_particle_tmp.vz, particle->vz, sizeof(float) * nParticles, cudaMemcpyHostToDevice);
    
    UpdateParticle<<<(nParticles+255)/256,256>>>(nParticles, gpu_particle, dt);
  
    cudaMemcpy(particle->x, gpu_particle_tmp.x, sizeof(float) * nParticles, cudaMemcpyDeviceToHost); 
    cudaMemcpy(particle->y, gpu_particle_tmp.y, sizeof(float) * nParticles, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle->z, gpu_particle_tmp.z, sizeof(float) * nParticles, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle->vx, gpu_particle_tmp.vx, sizeof(float) * nParticles, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle->vy, gpu_particle_tmp.vy, sizeof(float) * nParticles, cudaMemcpyDeviceToHost);
    cudaMemcpy(particle->vz, gpu_particle_tmp.vz, sizeof(float) * nParticles, cudaMemcpyDeviceToHost);
    
    // Move particles according to their velocities
    // O(N) work, so using a serial loop
    for (int i = 0 ; i < nParticles; i++) { 
        particle->x[i]  += particle->vx[i]*dt;
        particle->y[i]  += particle->vy[i]*dt;
        particle->z[i]  += particle->vz[i]*dt;
    }
}

void dump(int iter, int nParticles, struct ParticleArray * particle) {
    char filename[64];
    snprintf(filename, 64, "data/nbody/%s-%d.nbody", VERSION, iter);
    
    FILE * output;
    output = fopen(filename, "wb");

    fwrite(&initializer, sizeof(enum Initializer), 1, output);
    fwrite(&nParticles, sizeof(int), 1, output);
    fwrite(&iter, sizeof(int), 1, output);

    int i;
    for (i = 0; i < nParticles; i++)
    {
        fwrite(&particle->x[i], sizeof(float), 1, output);
        fwrite(&particle->y[i], sizeof(float), 1, output);
        fwrite(&particle->z[i], sizeof(float), 1, output);
        fwrite(&particle->vx[i], sizeof(float), 1, output);
        fwrite(&particle->vy[i], sizeof(float), 1, output);
        fwrite(&particle->vz[i], sizeof(float), 1, output);
    }

    fclose(output);
}

int main(const int argc, const char** argv)
{

  // Problem size and other parameters
  const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
  // Duration of test
  const int nSteps = (argc > 2)?atoi(argv[2]):10;
  // Particle propagation time step
  const float dt = 0.0005f;

    struct ParticleArray * particle = (struct ParticleArray *)malloc(sizeof(struct ParticleArray));
    particle->x = (float *)malloc(nParticles * sizeof(float));
    particle->y = (float *)malloc(nParticles * sizeof(float));
    particle->z = (float *)malloc(nParticles * sizeof(float));
    particle->vx = (float *)malloc(nParticles * sizeof(float));
    particle->vy = (float *)malloc(nParticles * sizeof(float));
    particle->vz = (float *)malloc(nParticles * sizeof(float));
    

  // Initialize random number generator and particles
  srand48(0x2020);

  int i;
  for (i = 0; i < nParticles; i++) {
     particle->x[i] =  2.0*drand48() - 1.0;
     particle->y[i] =  2.0*drand48() - 1.0;
     particle->z[i] =  2.0*drand48() - 1.0;
     particle->vx[i] = 2.0*drand48() - 1.0;
     particle->vy[i] = 2.0*drand48() - 1.0;
     particle->vz[i] = 2.0*drand48() - 1.0;
  }
  
  // Perform benchmark
  printf("\nPropagating %d particles using 1 thread...\n\n", 
	 nParticles
	 );
  double rate = 0, dRate = 0; // Benchmarking data
  const int skipSteps = 3; // Skip first iteration (warm-up)
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  for (int step = 1; step <= nSteps; step++) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    const double tStart = tv.tv_sec + tv.tv_usec / 1000000.0; // Start timing
    MoveParticles(nParticles, particle, dt);
    gettimeofday(&tv, NULL);
    const double tEnd =  tv.tv_sec + tv.tv_usec / 1000000.0; // End timing

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
    return EXIT_SUCCESS;
}


