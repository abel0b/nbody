#include <stdio.h>
#include <math.h>
#include <stdlib.h> // drand48
#include <omp.h>
#include <sys/time.h>

struct ParticleArray {
    float * x;
    float * y;
    float * z;
    float * vx;
    float * vy;
    float * vz; 
};

__global__ void UpdateParticle(const int nParticles, struct ParticleArray * const particle, const float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop over particles that experience force
    while (i < nParticles) {
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
        i += blockDim.x * gridDim.x;
    }
}

void MoveParticles(const int nParticles, struct ParticleArray * const particle, const float dt) {
    struct ParticleArray * gpu_particle;
    cudaMalloc(&gpu_particle, sizeof(struct ParticleArray));
    cudaMemcpy(particle, gpu_particle, sizeof(struct ParticleArray), cudaMemcpyHostToDevice);
  
    UpdateParticle<<<1,nParticles>>>(nParticles, gpu_particle, dt);
  
    cudaMemcpy(gpu_particle, particle, sizeof(struct ParticleArray), cudaMemcpyDeviceToHost);

    // Move particles according to their velocities
    // O(N) work, so using a serial loop
    for (int i = 0 ; i < nParticles; i++) { 
        particle->x[i]  += particle->vx[i]*dt;
        particle->y[i]  += particle->vy[i]*dt;
        particle->z[i]  += particle->vz[i]*dt;
    }
}

void dump(int iter, int nParticles, struct ParticleArray * particle)
{
    char filename[64];
    snprintf(filename, 64, "output_%d.txt", iter);

    FILE *f;
    f = fopen(filename, "w+");

    int i;
    for (i = 0; i < nParticles; i++)
    {
        fprintf(f, "%e %e %e %e %e %e\n",
                   particle->x[i], particle->y[i], particle->z[i],
		   particle->vx[i], particle->vy[i], particle->vz[i]);
    }

    fclose(f);
}

int main(const int argc, const char** argv)
{
    // Problem size and other parameters
    const int nParticles = (argc > 1 ? atoi(argv[1]) : 16384);
    // Duration of test
    const int nSteps = (argc > 2)?atoi(argv[2]):10;
    // Particle propagation time step
    const float dt = 0.0005f;

    struct ParticleArray particle;
    particle.x = (float *)malloc(nParticles * sizeof(float));
    particle.y = (float *)malloc(nParticles * sizeof(float));
    particle.z = (float *)malloc(nParticles * sizeof(float));
    particle.vx = (float *)malloc(nParticles * sizeof(float));
    particle.vy = (float *)malloc(nParticles * sizeof(float));
    particle.vz = (float *)malloc(nParticles * sizeof(float));

    // Initialize random number generator and particles
    srand48(0x2020);

    int i;
    for (i = 0; i < nParticles; i++) {
        particle.x[i] =  2.0*drand48() - 1.0;
        particle.y[i] =  2.0*drand48() - 1.0;
        particle.z[i] =  2.0*drand48() - 1.0;
        particle.vx[i] = 2.0*drand48() - 1.0;
        particle.vy[i] = 2.0*drand48() - 1.0;
        particle.vz[i] = 2.0*drand48() - 1.0;
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
    MoveParticles(nParticles, &particle, dt);
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
  
  free(particle.x);
  free(particle.y);
  free(particle.z);
  free(particle.vx);
  free(particle.vy);
  free(particle.vz);
}


