#include <iostream>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

using namespace std;

double time_diff(timeval a, timeval b){
    return (b.tv_sec-a.tv_sec)+pow(10,-6)*(b.tv_usec-a.tv_usec);
}


void init_array(double* ary, int s, bool random){
    for (int i=0; i<s; i++){
        if(random){
            ary[i] = (double)rand()/(double)(RAND_MAX);
        }
        else {
            ary[i] = 0.0;
        }
    }
}


void update_forces(double* ax,double* ay,double* az,double* xx,double* xy,double* xz, double* m, int i,int n){

    ax[i] = 0.0; 
    ay[i] = 0.0; 
    az[i] = 0.0; 

    for (int j=0;j<n; j++){
        if (j!=i){

            double dx = xx[j] - xx[i];
            double dy = xy[j] - xy[i];
            double dz = xz[j] - xz[i];

            double diff = dx*dx + dy*dy + dz*dz;
            double force = -6.67e-11 * (m[j]*m[i]) / diff;
        
            ax[i] += force * dx/m[i];
            ay[i] += force * dy/m[i];
            az[i] += force * dz/m[i];
        }
    }
}


void update_position(double* ax,double* ay,double* az,double* xx,double* xy,double* xz, double dt, int i){
    xx[i] += ax[i]*dt*dt;
    xy[i] += ay[i]*dt*dt;
    xz[i] += az[i]*dt*dt;
}


int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int n = atoi(argv[1]);  // number of objects

    double dt = 0.001;
    double total_time = 0.0;
    double used_time = 0.0;
    int steps = 100;

    timeval a;
    timeval b;

    //MPI_vars
    int rank, size;

    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //objects per process
    int per_proc = n/size + 1;

    srand(time(NULL));

    double* m  = new double[n];

    double* xx = new double[n];
    double* xy = new double[n];
    double* xz = new double[n];

    double* vx = new double[n];
    double* vy = new double[n];
    double* vz = new double[n];

    double* ax = new double[n];
    double* ay = new double[n];
    double* az = new double[n];

    //init initial values in rank=0 proc. 
    if(rank == 0){
        init_array(m,  n, true);
        init_array(xx, n, true);
        init_array(xy, n, true);
        init_array(xz, n, true);
        init_array(vx, n, false);
        init_array(vy, n, false);
        init_array(vz, n, false);
        init_array(ax, n, false);
        init_array(ay, n, false);
        init_array(az, n, false);
    }
    else{
        init_array(vx, n, false);
        init_array(vy, n, false);
        init_array(vz, n, false);
        init_array(ax, n, false);
        init_array(ay, n, false);
        init_array(az, n, false);
    }

    double* signal = new double[4]; 

    //send out init data
    if(rank == 0){
        for(int i=0; i<n; i++){
            for(int proc = 1; proc <size; proc++){
                //send init data to other procs
                signal[0] = m[i];
                signal[1] = xx[i];
                signal[2] = xy[i];
                signal[3] = xz[i];

                MPI_Send(signal, 4, MPI_DOUBLE, proc, i, MPI_COMM_WORLD);
            }
        }
    }
    else{

        //recv init data
        for (int proc = 1; proc < size; proc++){
            for (int i=0; i<n; i++){
                MPI_Recv(signal, 4, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &status);
                m[i] = signal[0];
                xx[i] = signal[1];
                xy[i] = signal[2];
                xz[i] = signal[3];
            }
        }
    }
            
    MPI_Barrier(MPI_COMM_WORLD);
    // init done

    //iter loop
    for (int iter = 0; iter < steps; iter ++){
        if(rank==0){
            gettimeofday(&a, NULL);
        }

        if(rank == 0){
            for(int i=0; i<n; i++){
                for(int proc = 1; proc <size; proc++){
                    signal[0] = xx[i];
                    signal[1] = xy[i];
                    signal[2] = xz[i];
                    MPI_Send(signal, 3, MPI_DOUBLE, proc, i, MPI_COMM_WORLD);
                }
            }
        }
        else{
            for (int proc = 1; proc < size; proc++){
                for (int i=0; i<n; i++){
                    MPI_Recv(signal, 3, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &status);
                    xx[i] = signal[0];
                    xy[i] = signal[1];
                    xz[i] = signal[2];
                }
            }
        }

        for(int i=0; i<n; i++){
            if(i%size==rank){
                update_forces(ax,ay,az, xx, xy,xz, m, i,n);
                update_position(ax,ay,az, xx,xy,xz, dt, i); 
                //do calc
            }
        }

        if (rank==0){
            for (int i=0;i<n;i++){
                if(i%size!=0){
                    MPI_Recv(signal,3,MPI_DOUBLE, i%size, i,MPI_COMM_WORLD,&status);
                    xx[i] = signal[0];
                    xy[i] = signal[1];
                    xz[i] = signal[2];
                }
            }
        }
        else {
            for(int i = 0; i<size; i++){
                if (i%size==rank){
                    signal[0] = xx[i];
                    signal[1] = xy[i];
                    signal[2] = xz[i];
                    MPI_Send(signal, 3, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
                }
            }
        }
                    
        
        if(rank==0){
            gettimeofday(&b, NULL);
            used_time+= time_diff(a,b);
        }
                
        MPI_Barrier(MPI_COMM_WORLD);
    }



    MPI_Finalize();

    return 0;
}









