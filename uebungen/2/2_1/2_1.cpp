#include <iostream>
#include <stdlib.h>
#include <mpi.h>

using namespace std; 

void print_time(double time, int rank, int m, bool verbose){
    if (verbose){
        cout<<rank<<": Timing: "<<time<<" total, \t"<<time/m<<" per msg."<<endl;
    }
    else {
        cout << time << " " << time/m <<  endl;
    }
}


int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    
    int rank, size, signal, m;      //MPI vars
    double a, b;                    //for timing
    double time = 0;                //total time 
    MPI_Status status;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    m = atoi(argv[1]);              //number msgs
    bool verbose = atoi(argv[2]);   //verbose output?


    //find next and previous process ranks
    int rank_n = (rank+1+1)%(size);
    int rank_p = (rank-1);
    if (rank_p < 0){
        rank_p = size-1;
    }

    //send msgs
    for (int i = 0; i< m; i++){
        a = MPI_Wtime();
        MPI_Send(NULL, 0, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD);
        b = MPI_Wtime();
        time += b-a;
        //cout << rank <<": Sent to " << rank_n << endl;
    }

    //recieve msgs, don't count to timing 
    for (int i = 0; i< m; i++){
        MPI_Recv(&signal, 0, MPI_INT, rank_p, 0, MPI_COMM_WORLD, &status);
        //cout << rank <<": Recv from " << rank_p << endl;
    }

    //report time to rank == 0 process
    if (rank == 0){
        double tmp_time;
        for (int i = 1; i < size; i++){
            MPI_Recv(&tmp_time, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            time += tmp_time;
        }
    }
    else{
        MPI_Send(&time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if(rank == 0){
        print_time(time, rank, m*size, verbose);
    }

    MPI_Finalize();

return 0;
}
