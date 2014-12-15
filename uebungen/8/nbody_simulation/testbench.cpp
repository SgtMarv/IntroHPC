#include <iostream>
using namespace std;

#include "nbody.h"
#include "coordinate.h"

#define MAX_OBJECTS 50
#define MAX_TIMESTEPS 10000
#define MAX_BODYMASS 1000
#define MIN_BODYMASS 100
#define XLENGTH 400
#define YLENGTH 400
#define ZLENGTH 400

int main(int argc, char* argv[])
{
	double pos[3];
	char *filename = NULL;

	int maxObjects = 0;
	int maxTimeSteps = 0;
	int maxBodyMass = 0;
	int minBodyMass = 0;

	if(argc > 1)	// parameter 1 -> filename
	{
		filename = new char[strlen(argv[1])+1];			// string filename erstellen
		strncpy(filename, argv[1], strlen(argv[1]));	// filename mit erstem parameter beschreiben
	}
	if(argc > 2)	// parameter 2 -> objects
	{
		maxObjects = atoi(argv[2]);
		if(maxObjects < 1)
		{
			cout << "wrong parameter 3 value: maxObjects need to be larger 0" << endl;
			return 0;
		}
	}
	if(argc > 3)	// parameter 3 -> timesteps
	{
		maxTimeSteps = atoi(argv[3]);
		if(maxTimeSteps < 1)
		{
			cout << "wrong parameter 4 value: maxTimeSteps need to be larger 0" << endl;
			return 0;			
		}
	}
	if(argc > 4)	// parameter 4 -> maxbodymass
	{
		maxBodyMass = atoi(argv[4]);
		if(maxBodyMass < 1)
		{
			cout << "wrong parameter 5 value: maxbodymass need to be larger 0" << endl;
			return 0;			
		}
	}
	if(argc > 5)	// parameter 5 -> minbodymass
	{
		minBodyMass = atoi(argv[5]);
		if(minBodyMass < 1)
		{
			cout << "wrong parameter 6 value: minbodymass need to be larger 0" << endl;
			return 0;			
		}
	}

	NBODY tempcords[maxObjects];	// temporärer Positionsvektor
	COORDINATE_SIZE cSize = {XLENGTH, XLENGTH, XLENGTH};

	cout << "start program: "
		 << "filename: " << filename << endl
		 << "maxObjects: " << maxObjects << endl
		 << "maxTimeSteps: " << maxTimeSteps << endl
		 << "maxBodyMass: " << maxBodyMass << endl
		 << "minBodyMass: " << minBodyMass << endl;
	
	// Aus der 'nbody.cpp/h Dateien'
	nbody3d simulation(maxObjects, maxBodyMass, minBodyMass, cSize);

	// WICHTIG!
	coordinate data(filename, maxObjects, maxTimeSteps, maxBodyMass, minBodyMass, cSize);

	// initializing Mass
	for(int i = 0; i < maxObjects; i++)
	{
		// WICHTIG
		data.initMass(i, simulation.getMass(i));
	}

	cout << "Programmstart..." << endl;


	for(int t = 0; t < maxTimeSteps; t++)
	{
		for(int i = 0; i < maxObjects; i++)
		{
			for(int j = 0; j < maxObjects; j++)
			{
				// Alle Kräfte die auf Object i einwirken
				simulation.calcForce(i, j);
			}
			// Neue Position für Object i wird berechnet
			simulation.calcPosition(i);
		}
		simulation.getPosition(tempcords, maxObjects);
		// WICHTIG
		data.writeTimestep(tempcords, maxObjects);
	}

	cout << "Programmende!" << endl;

	return 0;
}
