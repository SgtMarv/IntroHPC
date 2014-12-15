#include "nbody.h"

nbody3d::nbody3d(int numBodys, int maxmass, int minmass, COORDINATE_SIZE cSize)
{
	number_bodys = numBodys;
	maxMass = maxmass;
	minMass = minmass;
	cordSize.xLength = cSize.xLength;
	cordSize.yLength = cSize.yLength;
	cordSize.zLength = cSize.zLength;

	maxroom_size = 10.0;
	timestep = 0.0001;
	gravi = (double)1000.0;

	body = new SIMNBODY[number_bodys];

	cout << "creating nbodys...\n";
	srand((unsigned int)time(NULL));	// for random float positions
	for(int i = 0; i < number_bodys; i++)
	{
		initBody(i);
	}
}

nbody3d::~nbody3d()
{
	delete [] body;
}

int nbody3d::initBody(int bIndex)
{
	// body position between 0 - 100 (both axis)
	body[bIndex].position[X] = (((double)rand()/(double)(RAND_MAX)) * cordSize.xLength);
	body[bIndex].position[Y] = (((double)rand()/(double)(RAND_MAX)) * cordSize.yLength);
	body[bIndex].position[Z] = (((double)rand()/(double)(RAND_MAX)) * cordSize.zLength);
	// body mass between 100 - 1000
	body[bIndex].mass = (((double)rand()/(double)(RAND_MAX)) * this->maxMass) + this->minMass;
	return 1;
}

int nbody3d::calcDestination(int ObjIdx1, int ObjIdx2)
{
	// Entfernung in X- und Y-Koordinaten wird errechnet	
}

int nbody3d::calcForce(int ObjIdx1, int ObjIdx2)
{
	// Anziehungskraft zwischen ObjIdx1 und ObjIdx2 wird berechnet
}

int nbody3d::calcPosition(int ObjIdx)
{
	// Neue Position von ObjIdx wird berechnet
}

double nbody3d::getMass(int ObjIdx)
{
	return body[ObjIdx].mass;
}

int nbody3d::getPosition(int ObjIdx, double *vector)
{
	vector[X] = (double)body[ObjIdx].position[X];
	vector[Y] = (double)body[ObjIdx].position[Y];
	vector[Z] = (double)body[ObjIdx].position[Z];
	return 0;
}

int nbody3d::getPosition(NBODY* nbodys, int maxcircles)
{
	for(int i = 0; i < maxcircles; i++)
	{
		nbodys[i].circle = (int)i;
		nbodys[i].xCord = (double)body[i].position[X];
		nbodys[i].yCord = (double)body[i].position[Y];
		nbodys[i].zCord = (double)body[i].position[Z];
	} 

	return 0;
}
