#include <mpi.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>

using namespace std;

void Matrix_multiply(int *rank,int *size,int *numlines,int *numspalt)
{


 if(*(rank)==0)
{
    //Allokalisierung der MAtrizen Aund B
    int **Alines;
    int **Blines;
    int **Clines;

    Alines =(int **) malloc(*numlines*sizeof(int*));
    Blines =(int **) malloc(*numlines*sizeof(int*));
    Clines =(int **) malloc(*numlines*sizeof(int*));

    for(int i=0;i<*numlines;i++) Alines[i]= (int *) malloc(*numspalt*sizeof(int));
    for(int i=0;i<*numlines;i++) Blines[i]= (int *) malloc(*numspalt*sizeof(int));
    for(int i=0;i<*numlines;i++) Clines[i]= (int *) malloc(*numspalt*sizeof(int));

    //Initialieren der Matrizen A und B


    for(int i=0;i<*numlines;i++)
    {for(int j=0;j<*numspalt;j++)
     {Alines[i][j]=i+j;}}

    for(int i=0;i<*numlines;i++)
    {for(int j=0;j<*numspalt;j++)
     {Blines[i][j]=i*j;}}

    MPI_Status status;
    MPI_Request request;


	double starttime;
	double endtime;


	//starting measuremnt
	starttime=MPI_Wtime();

	//Verteilung der Matrizen
	for(int i=1;i<*size;i++)
	{
	for(int j=0;j<*numlines;j++)
	{
	        MPI_Isend(&Blines[j],1,MPI_INT,i,j,MPI_COMM_WORLD,&request);
	}}

	int lines1proc = *numlines/(*size-1);
	int ziel=1;

	for(int i=1;i<*numlines+1;i++)
	{
		if (i> lines1proc*ziel&&ziel<*size-1) ziel+=1;
		MPI_Isend(&Alines[i],1,MPI_INT,ziel,0,MPI_COMM_WORLD,&request);
	}


	//Empfangen der C-MAtrix
	for(int i=1;i<*size-1;i++)
	{
		for(int j=0;j<lines1proc;j++)
		{

		MPI_Recv(Clines[i*lines1proc+j],1,MPI_INT,i,j,MPI_COMM_WORLD,&status);
		}
	}	for (int j=0;j<lines1proc+(*numlines-(*size-1)*lines1proc);j++)
		{
		MPI_Recv(Clines[(*size-1)*lines1proc],1,MPI_INT,*size,j,MPI_COMM_WORLD,&status);
		}
	MPI_Barrier(MPI_COMM_WORLD);
	endtime=MPI_Wtime();

	//Ausgabe
	for (int i=0;i<*numlines;i++)
	{for (int j=0;j<*numspalt;j++)
		{ cout <<Clines[i][j]<<"\t";
		}
	cout <<endl;
	}
	cout <<"time "<<endtime-starttime<<endl;
	//de-allocalisierung der matrizen

    for(int i=0;i<*numlines;i++) free(Clines[i]);
    free(Clines);
    for(int i=0;i<*numlines;i++) free(Blines[i]);
    free(Blines);
    for(int i=0;i<*numlines;i++) free(Alines[i]);
    free(Alines);

}
//letzer slave
else if (*rank==*size){

	MPI_Status status;
	MPI_Request request;
	int *a[*numspalt];
	int **b[*numlines];
for(int i=0;i<*numlines;i++) int *b[i][*numspalt];
	int c[*numspalt];

    //MAtrix B empfange
    for(int i=0;i<*numlines;i++)
	{
		MPI_Recv(&b[i],1,MPI_INT,0,i,MPI_COMM_WORLD,&status);
	}

	//Daten matrix A empfangen : jeweiliger anteil plus rest
	for (int i=(*rank-1)*(*numlines/(*size));i<*numlines;i++)
	{
		MPI_Recv(&a,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
        //c berechnen
		for(int j=0;j<*numlines;j++)
		{
	       		 c[j]=0;
			for(int k=0;k<*numspalt;k++)
			{
				c[j]=*a[k]+(*b[j][k]);
			}
		}

	// zurücksenden

	MPI_Isend(c,1,MPI_INT,0,i,MPI_COMM_WORLD,&request);
	}



}
// alle anderen slaves
else {

	MPI_Status status;
	MPI_Request request;
	int *a[*numspalt];
	int **b[*numlines];
for(int i=0;i<*numlines;i++) int *b[i][*numspalt];
	int c[*numspalt];

	//Daten empfangen matrix B
	for(int i=0;i<*numlines;i++)
	{
		MPI_Recv(&b[i],1,MPI_INT,0,i,MPI_COMM_WORLD,&status);
	}

	//daten empgangen part of A
	for(int i=0;i<*numlines/(*size);i++)
	{
		MPI_Recv(&a,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
        //c berechnen

		for(int j=0;j<*numlines;j++)
		{	c[j]=0;
		for(int k=0;k<*numspalt;k++)
		{
			c[j]+=*a[k]+(*b[j][k]);
		}}

	//sending back
	MPI_Isend(c,1,MPI_INT,0,i,MPI_COMM_WORLD,&request);
	}
}
}

int main(int argc, char **argv)
{
int size,rank;
char hostname[50];
MPI_Init(&argc,&argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

int numspalt=5;
int numlines=5;

Matrix_multiply(&rank,&size,&numspalt,&numlines);


MPI_Finalize();
return 0;
}

