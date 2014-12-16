#ifndef NBODY_H_
#define NBODY_H_

#include <math.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
using namespace std;

#include "coordinate.h"

#define X 0
#define Y 1
#define Z 2

typedef struct
{
	double force[3];
	double position[3];
	double mass;
}SIMNBODY;

#ifndef CORDSIZE
#define CORDSIZE
typedef struct
{
	int xLength;
	int yLength;
	int zLength;
}COORDINATE_SIZE;
#endif


class nbody3d
{
private:
	SIMNBODY *body;
	COORDINATE_SIZE cordSize;
	int number_bodys;
	int maxMass;
	int minMass;
	double maxroom_size;

	double force[3];
	double timestep;	// dient der Integration von S = F * dt * dt
	double gravi; 	// Gravitationskonstante
	double R0_Vector[3];	// Nomierungsvektor
	double R2_Vector;	// Radius = Vektorbetrag(R0)

public:
	nbody3d(int numBodys, int maxmass, int minmass, COORDINATE_SIZE cSize);
	~nbody3d();

	int initBody(int bIndex);
	int calcDestination(int ObjIdx1, int ObjIdx2);
	int calcForce(int ObjIdx1, int ObjIdx2);
	int calcPosition(int ObjIdx);

	double getMass(int ObjIdx);
	int getPosition(int ObjIdx, double *vector);
	int getPosition(NBODY* nbodys, int maxcircles);

};

#endif