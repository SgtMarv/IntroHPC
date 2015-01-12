//
// simpleCUDA
//
// This simple code sample demonstrates how to perform a simple linear
// algebra operation using CUDA, single precision axpy:
// y[i] = alpha*x[i] + y[i] for x,y in R^N and a scalar alpha
//
// Please refer to the following article for detailed explanations:
// John Nickolls, Ian Buck, Michael Garland and Kevin Skadron
// Scalable parallel programming with CUDA
// ACM Queue, Volume 6 Number 2, pp 44-53, March 2008
// http://mags.acm.org/queue/20080304/
//
// Compilation instructions:
// - Install CUDA
// - Compile with nvcc -o simpleCUDA simpleCUDA.cu
// - Launch the executable
//
// 

#define DEBUG false
#define USE_PINNED_MEMORY true

#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

// CUDA imports (CUDA runtime, not necessary when compiling with nvcc)
//#include <cuda_runtime.h>


/////////////////////////////////////
// global variables and configuration section
/////////////////////////////////////

// problem size (vector length) N
long N;

// number of threads per block
long numThreadsPerBlock;

// device to use in case there is more than one
static int selectedDevice = 0;


/////////////////////////////////////
// kernel function (CPU)
/////////////////////////////////////
void saxpy_serial(int n, float alpha, float *x, float *y)
{
  int i;
  for (i=0; i<n; i++) {
    y[i] = alpha*x[i] + y[i];
  }
}


/////////////////////////////////////
// kernel function (CUDA device)
/////////////////////////////////////
__global__ void saxpy_parallel(int n, float alpha, float *x, float *y)
{
  // compute the global index in the vector from
  // the number of the current block, blockIdx,
  // the number of threads per block, blockDim,
  // and the number of the current thread within the block, threadIdx
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // except for special cases, the total number of threads in all blocks
  // adds up to more than the vector length n, so this conditional is
  // EXTREMELY important to avoid writing past the allocated memory for
  // the vector y.
  if (i<n) {
    y[i] = alpha*x[i] + y[i];
  }
}


/////////////////////////////////////
// error checking routine
/////////////////////////////////////
void checkErrors(char *label)
{
  // we need to synchronise first to catch errors due to
  // asynchroneous operations that would otherwise
  // potentially go unnoticed

  cudaError_t err;

  err = cudaThreadSynchronize();
  if (err != cudaSuccess) {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stdout, "CUDA Error: %s (at %s)\n", e, label);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stdout, "CUDA Error: %s (at %s)\n", e, label);
  }
}

void synchronize()
{
  cudaError_t err;
     
  err = cudaThreadSynchronize();
  if (err != cudaSuccess) {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at synchronize)\n", e);
  }
}

void printTime(char *label, timeval *tp1, timeval *tp2)
{
  //calculate and print out passed time measured with gettimeofday
  long usecs;

  usecs = (tp2->tv_sec - tp1->tv_sec) * 1E6 + tp2->tv_usec - tp1->tv_usec;
  // printf ("%s:\t\t%6ld usecs passed\n", label, usecs);
  printf ("%12ld\t", usecs);
}

/////////////////////////////////////
// main routine
/////////////////////////////////////
int main (int argc, char **argv)
{
  struct timeval tp1, tp2, tp3;
	long factor = 1; //for command line parsing
	char *pos = NULL;

  if (argc != 3) {
    printf ("Usage: %s <problem size{k,M,G}> <block size>\n", argv[0]);
    exit (0);
  }
  pos = strrchr (argv[1], 'k');
  if (pos != NULL) {
  	factor = 1024;
  	*pos = '\0'; //terminate input string here
  }
  pos = strrchr (argv[1], 'M');
  if (pos != NULL) {
  	factor = 1024*1024;
  	*pos = '\0'; //terminate input string here
  }
  pos = strrchr (argv[1], 'G');
  if (pos != NULL) {
  	factor = 1024*1024*1024;
  	*pos = '\0'; //terminate input string here
  }
  N = atol (argv[1]);
  N *= factor;
     
  //   the total number of blocks is obtained by rounding the
  //  vector length N up to the next multiple of numThreadsPerBlock
  numThreadsPerBlock = atol (argv[2]);
  int numBlocks = (N+numThreadsPerBlock-1) / numThreadsPerBlock;
     
  if (numThreadsPerBlock > 1024) {
    printf ("ERROR: numThreadsPerBlock must be <= 1024!\n");
    exit (0);
  }
  if (numBlocks >= 65536) {
    printf ("ERROR: numBlocks must be < 65536 (is %ld)!\n", numBlocks);
    exit (0);
  }

  /////////////////////////////////////
  // (1) initialisations:
  //     - perform basic sanity checks
  //     - set device
  /////////////////////////////////////
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "Sorry, no CUDA device fount");
    return 1;
  }
  if (selectedDevice >= deviceCount) {
    fprintf(stderr, "Choose device ID between 0 and %d\n", deviceCount-1);
    return 1;
  }
  cudaSetDevice(selectedDevice);
  checkErrors("initialisations");
  

  
  /////////////////////////////////////
  // (2) allocate memory on host (main CPU memory) and device,
  //     h_ denotes data residing on the host, d_ on device
  /////////////////////////////////////
  float *h_x;
  float *h_y;
  float *d_x;
  float *d_y;
  if (USE_PINNED_MEMORY) {
	printf ("Using pinned memory\n");
  	cudaMallocHost ( (void**) &h_x, N*sizeof(float) );
  	cudaMallocHost ( (void**) &h_y, N*sizeof(float) );
  }	else {
    h_x = (float*) malloc( N*sizeof(float) );
  	h_y = (float*) malloc( N*sizeof(float) );
  }
  cudaMalloc((void**)&d_x, N*sizeof(float));
  cudaMalloc((void**)&d_y, N*sizeof(float));
  checkErrors("memory allocation");

  //printf ("Running %s with problem size N=%ld and %ld threads per block (%d blocks total)\n", argv[0], N, numThreadsPerBlock, numBlocks);
  printf ("problem_size\tthreads_per_block\tnumBlocks\tinitData\tcudaMemcpy\tsaxpy_serial\tsaxpy_parallel_startup\tsaxpy_parallel_all\tcudaMemcpy\n");
  printf ("%12ld\t%12ld\t%12d\t", N, numThreadsPerBlock, numBlocks);



  /////////////////////////////////////
  // (3) initialise data on the CPU
  /////////////////////////////////////
  gettimeofday (&tp1, NULL);
  int i;
  for (i=0; i<N; i++) {
    h_x[i] = 1.0f + i;
    h_y[i] = (float)(N-i+1);
  }
  gettimeofday (&tp2, NULL);
  printTime ("initData", &tp1, &tp2);


  /////////////////////////////////////
  // (4) copy data to device
  /////////////////////////////////////
  gettimeofday (&tp1, NULL);
  cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);
  gettimeofday (&tp2, NULL);
  checkErrors("copy data to device");
  printTime ("cudaMemcpy", &tp1, &tp2);


  /////////////////////////////////////
  // (5) perform computation on host (to enable result comparison later)
  /////////////////////////////////////
  gettimeofday (&tp1, NULL);
  saxpy_serial(N, 2.0f, h_x, h_y);
  gettimeofday (&tp2, NULL);
  printTime ("saxpy_serial", &tp1, &tp2);


  /////////////////////////////////////
  // (6) perform computation on device
  //     - we use numThreadsPerBlock threads per block
  //     - the total number of blocks is obtained by rounding the
  //       vector length N up to the next multiple of numThreadsPerBlock
  /////////////////////////////////////
  gettimeofday (&tp1, NULL);
  saxpy_parallel<<<numBlocks, numThreadsPerBlock>>>(N, 2.0, d_x, d_y);
  gettimeofday (&tp2, NULL);
  synchronize(); //calculation is non-blocking
  gettimeofday (&tp3, NULL);
  printTime ("saxpy_parallel_startup", &tp1, &tp2);
  printTime ("saxpy_parallel_all", &tp1, &tp3);
  checkErrors("compute on device");



  /////////////////////////////////////
  // (7) read back result from device into temp vector
  /////////////////////////////////////
  float *h_z;
  if (USE_PINNED_MEMORY)
  	cudaMallocHost ( (void**) &h_z, N*sizeof(float) );
  else
  	h_z = (float*) malloc( N*sizeof(float) );

  gettimeofday (&tp1, NULL);
  cudaMemcpy(h_z, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  gettimeofday (&tp2, NULL);
  printTime ("cudaMemcpy", &tp1, &tp2);
  checkErrors("copy data from device");

  printf ("\n");
  
  /////////////////////////////////////
  // (8) perform result comparison
  /////////////////////////////////////
  int errorCount = 0;
  for (i=0; i<N; i++) {
    if (abs(h_y[i]-h_z[i]) > 1e-6) {
      errorCount = errorCount + 1;
      printf ("Mismatch: %lf vs %lf (pos %d)\n", h_y[i], h_z[i], i);
	}
    if (DEBUG) printf ("Output (cpu vs. gpu): %8.8lf vs %8.8lf (pos %d)\n", h_y[i], h_z[i], i);
  }
  if (errorCount > 0)
    printf("Result comparison failed.\n");



  /////////////////////////////////////
  // (9) clean up, free memory
  /////////////////////////////////////
	if (USE_PINNED_MEMORY) {
	  cudaFreeHost(h_x);
	  cudaFreeHost(h_y);
	  cudaFreeHost(h_z);
	} else {
	  free(h_x);
	  free(h_y);
	  free(h_z);
	}
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}
