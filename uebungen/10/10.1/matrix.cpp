#include <iostream>
#include <stdlib.h>

using namespace std;


void init_mat(float* mat, int size, bool zero){
    if(zero){
        for(int i = 0; i<(size*size); i++){
            mat[i] = 0.0;
        }
    } 
    else {
        for(int i = 0; i<(size*size); i++){
            mat[i] = (float)rand()/(double)(RAND_MAX);
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
    

int main(int argc, char** argv){

    int size;
    size = atoi(argv[1]);

    int seed = time(NULL);
    srand(seed);

    timeval start,stop;

    float* a = new float [size*size];
    float* b = new float [size*size];
    float* c = new float [size*size];

    init_mat(a,size,false);
    init_mat(b,size,false);
    init_mat(c,size,true);

    print_mat(a,size);
    print_mat(b,size);

    mat_mult_cpu(a,b,c,size);

    print_mat(c,size);


    return 0;
}
