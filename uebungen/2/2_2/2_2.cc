#include <mpi.h>
#include <iostream>
#include<stdlib.h>

using namespace std;

void barrier(int *rank,int *size)
{
 int signal=0;
 MPI_Status status;

 if(*(rank)!=0)
{
    //telling rank==0 im ready
    MPI_Send(&signal,1,MPI_INT,0,0,MPI_COMM_WORLD);
    //recieving from rank==0 everybody else is ready

    MPI_Recv(&signal,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
}
else{
        //waiting for all other threads
    for(int i=1;i<*(size);i++)
    {
    MPI_Recv(&signal,1,MPI_INT,i,0,MPI_COMM_WORLD,&status);
    }

        // telling all other threads that everybody is ready
    for(int i=1;i<*(size);i++)
    {
    MPI_Send(&signal,1,MPI_INT,i,0,MPI_COMM_WORLD);
    }


}
}

int main(int argc, char **argv)
{
int size,rank;
int num=atoi(argv[1]);
int numpro=atoi(argv[2]);
char hostname[50];
MPI_Init(&argc,&argv);

double starttime,endtime;
double starttime2,endtime2;


MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

starttime=MPI_Wtime();

// loop vor num of barriers
for(int i =0; i<num;i++)
{
    barrier(&rank,&size);
}


endtime=MPI_Wtime();

starttime2=MPI_Wtime();

for(int i=0;i<num;i++)
{
	MPI_Barrier(MPI_COMM_WORLD);	
}

endtime2=MPI_Wtime();
if (rank==0)
{ cout <<numpro<<"\t"<<endtime-starttime<<"\t"<<(endtime-starttime)/num<< "\t"<<endtime2-starttime2<<"\t"<<(endtime2-starttime2)/num<<endl;
}

MPI_Finalize();
return 0;
}

