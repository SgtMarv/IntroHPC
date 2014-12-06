#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <time.h>
using namespace std;



double cubic_abstand(double x1, double x2, double x3, double y1, double y2,double y3)
{
    double abstand=sqrt((x1-y1)*(x1-y1)+(x2-y2)*(x2-y2)+(x3-y3)*(x3-y3));
    return abstand*abstand*abstand;
}

int main (int argc, char** argv)
{
	N    = atoi(argv[1]);
 //Fixing the  constants
const double dt=0.001;
const double tmax=1;

 //MAssen
double m[N];
for(int i=0;i<N;i++) m[i]=1;

double t=0;

//arrays for position and velocities
double x[N][3]={0};
double xnew[N][3]={0};
double v[N][3]={0};
double vhalf=0;
double vnew[N][3]={0};

//zufallszahlen generator initialisieren
srand((unsigned)time(NULL));

//anfangsbedingen
 //Orte
    for (int i=0;i<N;i++)
    {
            x[i][0]=rand()*1./RAND_MAX;
            x[i][1]=rand()*1./RAND_MAX;
            x[i][2]=rand()*1./RAND_MAX;
    if (x[i][0]*x[i][0]+x[i][1]*x[i][1]+x[i][2]*x[i][2]>1)
       { i--;} //rejection
    }


//step counter
int s=0;
//time loop for integration
while(t<tmax)
{
    if(s%100==0)
    {
       ofstream out;
       char filename[20];
       snprintf (filename, sizeof filename, "data/2e/data_t_%f_.dat", t);
    out.open(filename);
 //ausgabe der werte
    for(int i=0;i<N;i++)
    {for(int j=0;j<3;j++)
        {out  <<"\t" <<x[i][j];}
    out <<endl;}
    out.close();
    }
    //Berechnung der neuen Geschwindigkeiten und Positionen (Leapfrog algorithm)

    //alle Richtungen
    for(int j=0;j<3;j++)
    {
        //alle Teilchen
        for(int i=0;i<N;i++)
        {
            vhalf=v[i][j];
            //alle anderen teilhen
            for(int k=0;k<N;k++)
            {if(k!=i){
                vhalf+=m[i]*m[k]*(x[k][j]-x[i][j])/cubic_abstand(x[k][0],x[k][1],x[k][2],x[i][0],x[i][1],x[i][2])*dt/2;
            }}

            xnew[i][j]=x[i][j]+vhalf*dt;
            vnew[i][j]=vhalf;

             //alle anderen teilhen
            for(int k=0;k<N;k++)
            {if(k!=i){
                vnew[i][j]+=m[i]*m[k]*(xnew[k][j]-xnew[i][j])/cubic_abstand(x[k][0],x[k][1],x[k][2],x[i][0],x[i][1],x[i][2])*dt/2;
            }}
        }
    }

    // Umspeichurung neue in alte Koordinaten

    for(int i=0;i<N;i++)
    {for(int j=0;j<3;j++)
        {
            x[i][j]=xnew[i][j];
            v[i][j]=vnew[i][j];
        }
    }

 t+=dt;
 s++;
}




}
