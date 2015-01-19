#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>


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
////////////////////////////////////////


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


}
    

int main(int argc, char** argv){

    int size;
    size = atoi(argv[1]);

    int seed = time(NULL);
    srand(seed);

    timeval start,stop;

    float* a = new float [size*size];
    float* b = new float [size*size];
    float* c = new float [size*size];

    init_mat(a,size,1);
    init_mat(b,size,2);
    init_mat(c,size,0);

    print_mat(a,size);
    print_mat(b,size);

    mat_mult_cpu(a,b,c,size);

    print_mat(c,size);


    return 0;
}
