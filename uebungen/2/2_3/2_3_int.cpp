#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

using namespace std;

void print_mat(int* mat, int size){
    if (size > 10){
        cout << "Matrix too large. Will not print." << endl;
    }
    else{
        for (int i = 0; i<(size*size);i++){
            cout << mat[i] << "\t";
            if ((i+1)%size == 0){
                cout << endl;
            }
        }
        cout << endl;
    }
}


void init_mat(int* mat, int size, bool zero){
    if (zero){
        for (int i=0; i<(size*size); i++){
            mat[i] = 0;
        }
    }
    else{
        for (int i=0; i<(size*size); i++){
            mat[i] = rand()%3+1;
        }
    }
}


void mat_mult(int* a, int* b, int* c,int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            for (int k = 0; k < size; k++){
                c[i*size+j] += a[i*size+k] * b[k*size+j];
            }
        }
    }
}


int main(int argc, char **argv){
    int size;                               
    size = atoi(argv[1]);
    int seed = time(NULL);
    srand(seed);

    timeval start,stop;                //for timing

    int* a = new int [size*size];
    int* b = new int [size*size];
    int* c = new int [size*size];


    init_mat(a, size, false);
    init_mat(b, size, false);
    init_mat(c, size, true);

    print_mat(a,size);
    print_mat(b,size);
    mat_mult(a,b,c,size);
    print_mat(c,size);


    return 0;
}
