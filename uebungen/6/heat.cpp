#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>


using namespace std; 


double time_diff(timeval a, timeval b){
    return (b.tv_sec-a.tv_sec)+pow(10,-6)*(b.tv_usec-a.tv_usec);
}


void print_init(double* grid, int n){   // debugg output 
    if (n<20){
        cout << endl;
        for (int i = 0; i<n; i++){
            for (int j = 0; j<n; j++){
                cout << grid[i*n+j] << "\t";
                if (j+1 == n){
                    cout << endl;
                }
            }
        }
        cout << endl;
    }
}

void init_grid(double* grid, int n){
    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            grid[i*n+j] = 0.0;
            if (i==0 && j>n/4 && j<n*3/4){  //init middle half
                grid[i*n+j] = 127.0;
            }
        }
    }
    print_init(grid,n);
}


double iteration(double* grid, int n){
    timeval a,b;
    gettimeofday(&a,NULL);

    // make copy of original grid
    double* x = new double[n*n];
    memcpy(x, grid, n*n*sizeof(double));

    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            if(!(i==0 || i+1==n || j==0 || j+1==n)){    // dont iterate border points
                //                                          orig       left           right         top             bottom
                grid[i*n+j]=x[i*n+j] + 0.24 * ((-4.0) * x[i*n+j] + x[i*n+(j-1)] + x[i*n+(j+1)] + x[(i-1)*n+j] + x[(i+1)*n+j]);
            }
        }
    }
    gettimeofday(&b, NULL);

    //cleanup
    delete [] x;
    x = NULL;

    return time_diff(a,b);
}


void print_state(double* grid, int n, int it){
    ofstream outfile;
    char name[255];
    sprintf(name,"./out/state%d.dat", it);
    outfile.open(name);
    for (int i = 0; i<n; i++){
        for (int j = 0; j<n; j++){
            outfile << j << " " << i << " " << grid[i*n+j] << endl;
        }
    }
    outfile.close();
}


int main (int argc, char** argv){

    MPI_Init(&argc, &argv);

    int rank, size; //MPI vars
    MPI_Comm cart_comm;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ndim = 1;   // MPI Cart vars
    int dim[ndim];
    int periods[ndim];
    int reorder = 0;
    dim[0] = size;
    periods[0] = 1;     //wrap around

    MPI_Cart_create(MPI_COMM_WORLD, ndim, dim, period, reorder, &cart_comm);

    int n    = 128; //default values can be over writen by cmd params
    int iter = 100;
    bool gif = false;
    double time_sum = 0.0;

    if (argc >= 4){
        n    = atoi(argv[1]);
        iter = atoi(argv[2]);
        gif  = atoi(argv[3]);
    }
    if (argc == 3){
        n    = atoi(argv[1]);
        iter = atoi(argv[2]);
    }
    if (argc == 2){
        n    = atoi(argv[1]);
    }

// done with init


    double* grid0 = new double [n*n];
    double* grid1 = new double [n*n];

    init_smile(grid,n);

    if(gif){
        print_state(grid,n,0);
    }
    for (int i = 1; i<=iter; i++){
        time_sum += iteration(grid,n);
        if(gif){
            print_state(grid,n,i);
        }
        cout << "ITeration: " << i << endl;
    }

    cout << "Total time: \t\t" << time_sum << endl;
    cout << "Time/Iteration: \t" << time_sum/iter << endl;
    cout << "GLOPS: \t\t" << (pow(10,-9) * n*n * 7)/(time_sum/iter) << endl;

    return 0;


