#include <mpi.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>

using namespace std;

void send(int *rank,int *i,double *time1,double *time2)
{
    //Nachricht zum Senden
 int size=pow(2,*i)*1000;
 int *signal;
 signal=(int *)malloc(size);
 int numpac=size/sizeof(int);

 MPI_Status status;
 MPI_Request request;

 if(*(rank)==0)
{	//measurement with blocking send

	double starttime;
	double endtime;

		starttime=MPI_Wtime();
    	MPI_Send(signal,numpac,MPI_INT,1,0,MPI_COMM_WORLD);
		endtime=MPI_Wtime();
		*time1+=(endtime-starttime);

	//measuremnt with nonblcking send


		starttime=MPI_Wtime();
    	MPI_Isend(signal,numpac,MPI_INT,1,0,MPI_COMM_WORLD,&request);
		endtime=MPI_Wtime();
		*time2+=endtime-starttime;


}
else if (*rank==1){



    MPI_Recv(signal,numpac,MPI_INT,0,0,MPI_COMM_WORLD,&status);





    MPI_Irecv(signal,numpac,MPI_INT,0,0,MPI_COMM_WORLD,&request);


}

}

int main(int argc, char **argv)
{
int size,rank;
int num=atoi(argv[1]); //how many sizes
int iter=atoi(argv[2]); //how many iterartions for one size
char hostname[50];
MPI_Init(&argc,&argv);

double time_block;
double time_nonblock;

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);



// loop vor num of sends with different sizes

for(int i =0; i<num;i++)
{
    time_block=0;
    time_nonblock=0;

    //loopp for all iterations for one size
    for(int j=0;j<iter;j++)
    {

    send(&rank,&i,&time_block,&time_nonblock);

    }
	if(rank==0)
	{
	    //werte ausgeben
	cout << i << "\t" << pow(2,i)<< "\t"<<iter*pow(2,i)*1000/time_block<<"\t"<<iter*pow(2,i)*1000/time_nonblock<<endl;
	}
}

MPI_Finalize();
return 0;
}

