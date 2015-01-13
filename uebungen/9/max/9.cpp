#include <iostream> 
#include <stdlib.h>
#include <sys/time.h>
#include <cstring>
#include <math.h>

using namespace std;

float time_diff(timeval a, timeval b){
    return (b.tv_sec-a.tv_sec) * pow(10,-6) * (b.tv_usec - a.tv_usec);
}

void init_array(float* ary, int n){
    for(int i = 0; i<n i++){
        ary[i] = (float)rand()/(float(RAND_MAX);
    }
}



void saxpy_cpu(float* x, float* y, float a, int n){
    for(int i=0; i<n; i++){
        y[i] = a * x[i] + y[i];
    }
}


__global__ void saxpy_gpu(float* x, float* y, float a, int n){

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if ( i<n ) {
        y[i] = a * x[i] + y[i];
    }

}


int main (int argc, char **argv){

    long n;      //problem size
    float a = 2.0; //alpha factor in equation, hardcoded
    bool pinned_mem; //use pinned memory or not

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
  n = atol (argv[1]);
  n *= factor;
     
  long numThreadsPerBlock;
  int selectedDevice = 0;
  
  numThreadsPerBlock = atol (argv[2]);
  int numBlocks = (N+numThreadsPerBlock-1) / numThreadsPerBlock;
     
  if (numThreadsPerBlock > 1024) {
    printf ("ERROR: numThreadsPerBlock must be <= 1024!\n");
    return 0;
  }
  if (numBlocks >= 65536) {
    printf ("ERROR: numBlocks must be < 65536 (is %ld)!\n", numBlocks);
    return 0;
  }

    //for timing
    timeval start, stop;
    float t_init; 
    float t_copy;
    float t_cpu;
    float t_gpu;
    float t_back;


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
  cudaThreadSynchronize();


    //allcoate mem
    float* x;
    float* y;
    float* d_x;
    float* d_y;

    //where to alloc host vars
    if (pinned_mem){
        cudaMallocHost((void**) &x, N*sizeof(float));
        cudaMallocHost((void**) &y, N*sizeof(float));
    }
    else {
        x = (float*) malloc(N*sizeof(float));
        y = (float*) malloc(N*sizeof(float));
    }

    //allocate device vars on GPU 
    cudaMalloc((void**)&d_x, N*sizeof(float));
    cudaMalloc((void**)&d_y, N*sizeof(float));

   
    //init arrays on CPU
    gettimeofday(&start, NULL);
    init_array(x, n);
    init_array(y, n);
    gettimeofday(&stop, NULL);
    t_init = diff_time(start,stop);

    
    // copy to GPU
    gettimeofday(&start, NULL);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    gettimeofday(&stop, NULL);
    t_copy = diff_time(start, stop);

    // do operation on CPU 
    gettimeofday(&start, NULL);
    saxpy_cpu(x, y, a, N);
    gettimeofday(&stop, NULL);
    t_cpu = time_diff(start, stop);

    // do operation on GPU
    gettimeofday(&start, NULL);
    saxpy_cpu<<<numBlocks, numThreadsPerBlock>>>(d_x, d_y, a, N);
    cudaThreadSynchronize();
    gettimeofday(&stop, NULL);
    t_gpu = time_diff(start, stop);

    
    //write back and compare
    float* tmp_y; 
    if (pinned_mem){
        cudaMallocHost((void**) &tmp_y, N*sizeof(float));
    }
    else {
        tmp_y = (float*) malloc(N*sizeof(float));
    }

    gettimeofday(&start, NULL);
    cudaMemcpy(tmp_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    gettimeofday(&stop, NULL);
    t_back = time_diff(start, stop);

    int err_count = 0;
    for(int i = 0; i<N; i++){
        if(abs(tmp_y[i]-y[i]) > 1e-6){
            cout << "Error on comparison on index: " << i << endl;
            err_count++;
        }
    }
    cout << "Error count: " << err_count << endl;


    //cleanup
    if(pinned_mem){
        cudaFreeHost(x);
        cudaFreeHost(y);
        cudaFreeHost(tmp_y);
    }
    else{
        free(x);
        free(y);
        free(tmp_y);
    }

    cudaFree(d_x);
    cudaFree(d_y);

    //report timing
    cout << "Initialization: " << t_init << " s\n";
    cout << "Copy to GPU: " << t_copy << " s\n";
    cout << "Sequential CPU: " << t_cpu << " s\n";
    cout << "Parallel GPU: " << t_gpu << " s\n";
    cout << "Writeback: " << t_back << " s\n";


    return 0;



}







