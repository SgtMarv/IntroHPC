#include <iostream> 
#include <stdlib.h>
#include <sys/time.h>
#include <cstring>
#include <math.h>

using namespace std;

double time_diff(timeval a, timeval b){
    return (b.tv_sec-a.tv_sec) * pow(10,-6) * (b.tv_usec - a.tv_usec);
}


void saxpy_cpu(double* x, double* y, double a, int n){
    for(int i=0; i<n; i++){
        y[i] = a * x[i] + y[i];
    }
}

int main (int argc, char **argv){


    return 0;



}

