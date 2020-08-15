#include <vector>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <curand.h>
#include <time.h>       /* time */
#include <random>


float value(   
  curandGenerator_t curandGenerator, 
  float *d_normals, 
  float *d_s, 
  size_t N_PATHS,
  size_t N_STEPS,
  const float K,
  const float B,
  const float S0,
  const float sigma ,
  const float mu,
  const float r ) ;

  float pickOne( float rng, float *v ) ;


#define CHECKCURAND(expression)                         \
  {                                                     \
    curandStatus_t status = (expression);                         \
    if (status != CURAND_STATUS_SUCCESS) {                        \
      std::cerr << "Curand Error on line " << __LINE__<< std::endl;     \
      std::exit(EXIT_FAILURE);                                          \
    }                                                                   \
  }

// atomicAdd is introduced for compute capability >=6.0
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
      printf("device arch <=600\n");
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
          unsigned long long int old = *address_as_ull, assumed;
            do {
                    assumed = old;
                        old = atomicCAS(address_as_ull, assumed,
                                                    __double_as_longlong(val + __longlong_as_double(assumed)));
                          } while (assumed != old);
              return __longlong_as_double(old);
}
#endif

__global__ void sumPayoffKernel(float *d_s, const unsigned N_PATHS, double *mysum)
{
  unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned stride = blockDim.x * gridDim.x;
  unsigned tid = threadIdx.x;

  extern __shared__ double smdata[];
  smdata[tid] = 0.0;

  for (unsigned i = idx; i<N_PATHS; i+=stride)
  {
    smdata[tid] += (double) d_s[i];
  }

  for (unsigned s=blockDim.x/2; s>0; s>>=1)
  {
    __syncthreads();
    if (tid < s) smdata[tid] += smdata[tid + s];
  }

  if (tid == 0)
  {
    atomicAdd(mysum, smdata[0]);
  }
}

__global__ void barrier_option(
    float *d_s,
    const float T,
    const float K,
    const float B,
    const float S0,
    const float sigma,
    const float mu,
    const float r,
    const float * d_normals,
    const long N_STEPS,
    const long N_PATHS)
{
  unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned stride = blockDim.x * gridDim.x;
  const float tmp1 = mu*T/N_STEPS;
  const float tmp2 = exp(-r*T);
  const float tmp3 = sqrt(T/N_STEPS);
  double running_average = 0.0;

  for (unsigned i = idx; i<N_PATHS; i+=stride)
  {
    float s_curr = S0;
    for(unsigned n = 0; n < N_STEPS; n++){
       s_curr += tmp1 * s_curr + sigma*s_curr*tmp3*d_normals[i + n * N_PATHS];
       running_average += (s_curr - running_average) / (n + 1.0) ;
       if (running_average <= B){
           break;
       }
    }

    float payoff = (running_average>K ? running_average-K : 0.f);
    d_s[i] = tmp2 * payoff;
  }
}

int main(int argc, char *argv[]) {
  srand48(time(NULL) );
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{0.5,.25};

  try {
    // declare variables and constants
    size_t N_PATHS = 4096000;
    size_t N_STEPS = 365;
    if (argc >= 2)  N_PATHS = atoi(argv[1]);
    if (argc >= 3)  N_STEPS = atoi(argv[2]);

    const size_t N_NORMALS = (size_t)N_STEPS * N_PATHS;

    // Generate random numbers on the device
    curandGenerator_t curandGenerator;
    CHECKCURAND(curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32));
    float *d_normals;
    float *d_s;
    checkCudaErrors(cudaMalloc(&d_normals, N_NORMALS * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_s, N_PATHS*sizeof(float)));

    std::cout 
        << "Price" << ","
        << "K" << ","
        << "B" << ","
        << "S0" << ","
        << "sigma" << ","
        << "mu" << ","
        << "r" << std::endl ;

    // float K = 110.0f;
    // float B = 100.0f;
    // float S0 = 120.0f;
    // float sigma = 0.35f;
    // float mu = 0.1f;
    // float r = 0.05f;

    float Kmnmx[] =     {-20.f, 20.f } ;     // Strike
    float Bmnmx[] =     {0.f, 20.f } ;     // {0, 198.f} ;    // Barrier
    float S0mnmx[] =    {0, 200.f} ;    // Spot
    float sigmamnmx[] = {0, 0.4f} ;     // volatility %
    float mumnmx[] =    {0, 0.2f} ;     // drift
    float rmnmx[] =     {0, 0.2f} ;     // risk free rate

    for( int i=0 ; i<20000 ; i++ ) {
      
      float S0 = pickOne( d(gen), S0mnmx ) ; 
      float K = S0 - pickOne( d(gen), Kmnmx ) ;
      float B = min( S0,K ) - pickOne( d(gen), Bmnmx ) ; 
      float sigma = pickOne( drand48(), sigmamnmx ) ; 
      float mu = pickOne( drand48(), mumnmx ) ; 
      float r = pickOne( drand48(), rmnmx ) ; 

      float val = value( curandGenerator, d_normals, d_s, N_PATHS, N_STEPS, K, B, S0, sigma, mu, r ) ;

      std::cout 
          << val << ","
          << K << ","
          << B << ","
          << S0 << ","
          << sigma << ","
          << mu << ","
          << r
      << std::endl ;
    }
    CHECKCURAND(curandDestroyGenerator( curandGenerator )) ;
    checkCudaErrors(cudaFree(d_s));
    checkCudaErrors(cudaFree(d_normals));
   
  } catch(std::exception& e) {
    std::cerr<< "exception: " << e.what() << "\n";
  }
}

float pickOne( float x, float *v ) {
  if( x<v[0] ) x=v[0] ;
  if( x>v[1] ) x=v[1] ;
  return v[0] + ( ( v[1] - v[0] ) * x )  ;
}


float value(     
          curandGenerator_t curandGenerator,
          float *d_normals, 
          float *d_s, 
          size_t N_PATHS,
          size_t N_STEPS,
          const float K,
          const float B,
          const float S0,
          const float sigma ,
          const float mu,
          const float r ) {

  const float T = 1.0f;
  double gpu_sum{0.0};

  int BLOCK_SIZE, GRID_SIZE ;
  checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&GRID_SIZE,
                                                     &BLOCK_SIZE,
                                                     barrier_option,
                                                     0, N_PATHS));

  const size_t N_NORMALS = (size_t)N_STEPS * N_PATHS;

  CHECKCURAND(curandGenerateNormal(curandGenerator, d_normals, N_NORMALS, 0.0f, 1.0f));
  cudaDeviceSynchronize();

  barrier_option<<<GRID_SIZE, BLOCK_SIZE>>>(d_s, T, K, B, S0, sigma, mu, r, d_normals, N_STEPS, N_PATHS);
  cudaDeviceSynchronize();

  double* mySum;
  checkCudaErrors(cudaMallocManaged(&mySum, sizeof(double)));
  sumPayoffKernel<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE*sizeof(double)>>>(d_s, N_PATHS, mySum);
  cudaDeviceSynchronize();

  gpu_sum = mySum[0] / N_PATHS;

  checkCudaErrors(cudaFree(mySum));

  return gpu_sum ;
}