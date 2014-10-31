#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

using namespace std;

void print_mat(double* mat, int size){
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


void init_mat(double* mat, int size, bool zero){
    if (zero){
        for (int i=0; i<(size*size); i++){
            mat[i] = 0;
        }
    }
    else{
        for (int i=0; i<(size*size); i++){
            mat[i] = (double)rand()/(double)(RAND_MAX);
        }
    }
}


void mat_mult(double* a, double* b, double* c,int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            for (int k = 0; k < size; k++){
                c[i*size+j] += a[i*size+k] * b[k*size+j];
            }
        }
    }
}


int main(int argc, char **argv){
    int size, runs; 
    size = atoi(argv[1]);
    runs = atoi(argv[2]);
    
    int seed = time(NULL);
    srand(seed);

    timeval start,stop;                //for timing

    double* a = new double [size*size];
    double* b = new double [size*size];
    double* c = new double [size*size];


    for(int i = 0; i < runs; i++){
        init_mat(a, size, false);
        init_mat(b, size, false);
        init_mat(c, size, true);

        //print_mat(a,size);
        //print_mat(b,size);
        gettimeofday(&start,NULL);
        mat_mult(a,b,c,size);
        gettimeofday(&stop,NULL);
        //print_mat(c,size);

        double time= (stop.tv_sec-start.tv_sec) + pow(10,-6)*(stop.tv_usec-start.tv_usec);

        cout << "Time: " << time << "s\n";
    }
  

    //cleanup foo
    delete[] a;
    delete[] b;
    delete[] c;
    a = NULL;
    b = NULL;
    c = NULL;


    return 0;
}
