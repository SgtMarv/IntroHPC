#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

using namespace std; 

void fill_array(double* ary, int s){
    srand(time(NULL));
    for (int i = 0; i<s; i++){
        ary[i] = (double)rand()/(double)(RAND_MAX);
    }
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
   
    int msg_size = 10;              //from 2**0 to 2**msg_size KB of data
    int msg_count = 10;             //number of mesages per iter

    double a, b, time;              //for timing
    double time_gnuplot[msg_size+1][msg_count];  //for nice formating 

    int rank, size, signal;         //MPI vars
    MPI_Status status;	
    MPI_Request request;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    //start latency loops
    for (int j = 0; j<=msg_size; j++){      // outer loop over mesage size
        int size = (int)pow(2,j);           // 2**j KB of data
        double* signal = new double [size*16]; //16 double = 1KB
        if (rank ==0){                      // fill array with random doubles 
            fill_array(signal,size*16);
        }

        for (int i = 0; i<msg_count; i++){ //inner loop for number of messages
            if (rank == 0){
                time = 0.0;
                a = MPI_Wtime();
                MPI_Isend(signal, size*16, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,&request);
                MPI_Recv(signal, size*16, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD,&status);
                b = MPI_Wtime();
                time += b-a;
                time_gnuplot[j][i] = time/2.0;    //half round trip latency
            }
            else{
                if (rank == 1){   //avoid starvation if more than two proc
                    MPI_Recv(signal,size*16,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&status);
                    MPI_Isend(signal,size*16,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&request);
                }
            }
        }//end of mesage count loop
        delete[] signal;
        signal = NULL;
    }//end of size loop

    for (int j=0; j<=msg_size; j++){
        for (int i=0; i<msg_count;i++){
            cout << time_gnuplot[j][i] << " ";
            if (i==msg_count-1){
                cout << "\n";
            }
        }
    }




    MPI_Finalize();
    return 0;
}
