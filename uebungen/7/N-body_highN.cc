#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <time.h>
using namespace std;

void print_pos(int *N,double ***x,double *t)
{
     ofstream out;
       char filename[20];
       snprintf (filename, sizeof filename, "data/2e/data_t_%f_.dat", *t);
    out.open(filename);
 //ausgabe der werte
    for(int i=0;i<*N;i++)
    {for(int j=0;j<3;j++)
        {out  <<*x[i][j]<<"\t";}
    out <<endl;}
    out.close();
}



void accelarations(int *N, double ***x,double ***a,double **m)
{
    //Initialisieren
    for(int i=0;i<*N;i++)
    {for(int j=0;j<3;j++){(*a)[i]=0;}}

    //schleife über alle teilchen paare
    for(int i=0;i<*N;i++)
    {
        for(int j=i+1;j<*N;j++)
        {

                 double dx=(*x)[j][0]-(*x)[i][0];
                double dy=(*x)[j][1]-(*x)[i][1];
                double dz=(*x)[j][2]-(*x)[i][2];

                double diff= dx*dx+dy*dy+dz*dz;
                double kraft=-6.673e-11*(*m)[i]*(*m)[j]/diff;


                (*a)[i][0]+=kraft*dx/(*m)[i];
                (*a)[i][1]+=kraft*dy/(*m)[i];
                (*a)[i][2]+=kraft*dz/(*m)[i];

                //kraft und gegenkraft, da keine interaktion doppelt gezählt wird
                (*a)[j][0]-=kraft*dx/(*m)[j];
                (*a)[j][1]-=kraft*dy/(*m)[j];
                (*a)[j][2]-=kraft*dz/(*m)[j];

        }
    }

}

void pos_update(int *N,double dt,double (***x),double(***a))
{
     //Teilchen
    for(int i=0;i<*N;i++)
    {   //Richtungen
        for(int j=0;j<3;j++)
        {
            (*x)[i][j]+=(*a)[i][j]*(dt)*(dt);
        }
    }
}

int main (int argc, char** argv)
{
int	N    = atoi(argv[1]);
 //Fixing the  constants
const double dt=0.001;
const int steps=1000;

//time varialbe
double t=0;

//arrays for position and beschleunigung und massen
    double **x;
    double **a;
    double *m;
    x=(double **) malloc(N*sizeof(double*));
    for(int i=0;i<3;i++) x[i]= (double *) malloc(3*sizeof(double));
    a=(double **) malloc(N*sizeof(double*));
    for(int i=0;i<3;i++) a[i]= (double *) malloc(3*sizeof(double));
    m=(double *) malloc(N*sizeof(double));

    //zufallszahlen generator initialisieren
    srand((unsigned)time(NULL));

    //anfangsbedingen
    //Orte
    for (int i=0;i<N;i++)
    {for(int j=0;j<3;j++)
            x[i][j]=rand()*1./RAND_MAX;
    }

    //accelarations
    for (int i=0;i<N;i++)
    {for(int j=0;j<3;j++)
            a[i][j]=0;
    }

    //Massen
   for (int i=0;i<N;i++)
    {
            m[i]=rand();
    }


    //step counter
    int s=0;
    //time loop for integration
    while(s<steps)
    {
        //Ausgabe
        if(s%100==0)
        { print_pos(&N,&x,&t); }


        //Berechnung der neuen Beschleunigungen
        accelarations(&N,&x,&a,&m);

       //Berechnung der nuene Orte
       pos_update(&N,dt,&x,&a);


    t+=dt;
    s++;
}

free(x);
free(a);
free(m);


}
