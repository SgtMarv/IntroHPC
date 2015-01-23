#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>


using namespace std;

static int selectedDevice = 0;
long numThreadsPerBlock;

//copy from ex9 file
void checkErrors(char *label)
{
  cudaError_t err;

  err = cudaThreadSynchronize();
  if (err != cudaSuccess) {
    char *e = (char*) cudaGetErrorString(err);
    cout << "Cuda Error: " << e << " (at " << label <<")" << endl;
  }

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    char *e = (char*) cudaGetErrorString(err);
    cout << "Cuda Error: " << e << " (at " << label <<")" << endl;
  }
}

void synchronize()
{
  cudaError_t err;
     
  err = cudaThreadSynchronize();
  if (err != cudaSuccess) {
    char *e = (char*) cudaGetErrorString(err);
    cout << "Cuda Error: " << e << " (at synchronize)" << endl;
  }
}
////////////////////////////////////////

double time_diff(timeval a, timeval b){
    return (b.tv_sec-a.tv_sec)+pow(10,-6)*(b.tv_usec-a.tv_usec);
}


void init_mat(float* mat, int size, int init){
    if(init==0){
        for(int i = 0; i<(size*size); i++){
            mat[i] = 0.0;
        }
    } 
    if(init == 1){
        for(int i = 0; i<size; i++){
            for(int j = 0; j<size; j++){
                mat[i*size+j] = (i+1)+(j+1);
            }
        }
    }
    if(init == 2){
        for(int i = 0; i<size; i++){
            for(int j = 0; j<size; j++){
                mat[i*size+j] = (i+1)*(j+1);
            }
        }
    }
}

void print_mat(float* mat, int size){
    if(size>10){
        cout << "Matrix too large, will not print." << endl;
    }
    else{
        for(int i = 0; i<(size*size); i++){
            cout << mat[i] << "\t";
            if((i+1)%size == 0){
                cout << endl;
            }
        }
        cout << endl;
    }
}

void mat_mult_cpu(float* a, float* b, float* c, int size){
    for (int i = 0; i<size; i++){
        for (int k = 0; k<size; k++){
            for (int j = 0; j<size; j++){
                c[i*size+j] += a[i*size+k] * b[k*size+j];
            }
        }
    }
}

__global__ void mat_mult_gpu(float* a, float* b, float* c, int size){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float pval = 0.0;
    if(row<size&&col<size){
for (int k = 0; k< size; k++){
        pval += a[row * size + k] * b[k * size + col];
    }

    c[row*size+col] = pval;
}
}

__global__ void mat_mult_shared(float* a, float* b, float* c, int size){

	__shared__ float mem_ds [];

	 float *a_s=&(mem_ds [0]);
	float *b_s=&(mem_ds [size*size]);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float pval = 0.0;
    if(row<size&&col<size){
	  for ( int m = 0; m < size; ++m ) {
     // Collaborative loading of a and b tiles into shared memory
     a_s [row*size+col] = a [ row * size + ( m * size + col ) ];
	 b_s [row*size + col] = b [ col + ( m * size	 + row ) * size ];
 __syncthreads();
 for ( int k = 0; k < size; ++k )
 	pval += a_s[row*size+k] * b_s[k*size+col];
 __syncthreads ();
 }
     c[row*size+col] = pval;
}
}
    

int main(int argc, char** argv){

    int size;
    float t_send;
    size = atoi(argv[1]);

    numThreadsPerBlock = atol(argv[2]);
    int numBlocks = 1;//(size+numThreadsPerBlock-1)/numThreadsPerBlock;
    if (numThreadsPerBlock > 1024){
        cout << "ERROR: NumThreadPerBlock must be < 1024" << endl;
        return 0;
    }
    if(numBlocks >65536){
        cout << "ERROR: numBlocks must be < 65536, is " << numBlocks << endl;
        return 0;
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount== 0){
        cout << "ERROR: No device found" << endl;
        return 0;
    }
    if(selectedDevice >= deviceCount){
        cout << "ERROR: Choose dev ID between 0 and " << deviceCount-1 << endl;
        return 0;
    }
    cudaSetDevice(selectedDevice);
    checkErrors("init");


    int seed = time(NULL);
    srand(seed);

    timeval start,stop;

    float* h_a = new float [size*size];
    float* h_b = new float [size*size];
    float* h_c = new float [size*size];
    float* c_comp = new float [size*size];

    float* d_a = new float [size*size];
    float* d_b = new float [size*size];
    float* d_c = new float [size*size];

    init_mat(h_a,size,1);
    init_mat(h_b,size,2);
    init_mat(h_c,size,0);

    cudaMalloc((void**) &d_a, size*size*sizeof(float));
    cudaMalloc((void**) &d_b, size*size*sizeof(float));
    cudaMalloc((void**) &d_c, size*size*sizeof(float));
    checkErrors("mem alloc");

    gettimeofday(&start,NULL);	
    cudaMemcpy(d_a, h_a, size*size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size*size*sizeof(float), cudaMemcpyHostToDevice);
    gettimeofday(&stop,NULL);
    t_send=time_diff(start,stop);
    checkErrors("copy date to dev");
   

    print_mat(h_a,size);
    print_mat(h_b,size);

    //mat mult CPU
    gettimeofday(&start,NULL);
    mat_mult_cpu(h_a,h_b,h_c,size);
    gettimeofday(&stop,NULL);
    cout << "Time for CPU: " << time_diff(start,stop) << " s" << endl;

    dim3 dimBlock(size,size);
    //mat mult GPU
    gettimeofday(&start,NULL);
    mat_mult_gpu<<<numBlocks,dimBlock>>> (d_a,d_b,d_c,size);
    gettimeofday(&stop,NULL);
    synchronize();
    cout << "Time for GPU: " << time_diff(start,stop) << " s" << endl;
    checkErrors("compute on GPU");

   //mat mult with shared mem on gpu
	int size_block=size;
 	int shared_size=2*size_block*size_block*sizeof(float);
	gettimeofday(&start,NULL);

	mat_mult_shared<<numBlocks,dimBlock,shared_size>> (d_a,d_b,d_c,size);
	gettimeofday(&stop,NULL);
  	synchronize();
	cout <<"Time with shared memory: "<< time_diff(start,stop)<<endl;	
	checkErrors("compute on GPU_shared");

    gettimeofday(&start,NULL);
    cudaMemcpy(c_comp, d_c, size*size*sizeof(float), cudaMemcpyDeviceToHost); 
    gettimeofday(&stop,NULL);    
    t_send+=time_diff(start,stop);
    checkErrors("copy back");
    cout <<"Time for Sending: "<<t_send<<endl;

    print_mat(h_c,size);
	print_mat(c_comp,size);
	for(int i=0;i<size;i++){
	for(int j=0;j<size;j++){
		if(h_c[i*size+j]!=c_comp[i*size+j])cout<<"e";}}
	
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
   
    free(h_a);
    free(h_b);
    free(h_c);
    free(c_comp);

    return 0;
}
