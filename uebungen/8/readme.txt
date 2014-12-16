nbody_visualisation: Visualisierungs Programm.
	Muss auf dem Rechner 'ceg01' aufgerufen werden.
	
	user@creek01:~/$ ssh -XC username@ceg01	(Wichtig: Man sollte bereits vom eigenen Rechner aus mit ssh -X ... eingeloggt sein)
	user@ceg01:~/$ cd /home/erusakov/HPC/Exercise09/
	user@ceg01:~/$ ./nbody_visualisation
	
	CvmIF.jpg im Ordner lassen, das Programm braucht es um die Farbskala darzustellen.

nbody_simulation Ordner:
Enthält 'main' als simulations programm. 'main' kann über die Konsole oder über 'run.sh' ausgeführt werden. 
Parameter 'main': main [filename] [number of nbodys] [timesteps] [maxmass] [minmass]
Beispiel für data.bin, 100 nbodys, 10000 timesteps, 1000 maxmass und 100 minmass

	user@creek01:~/$ ./main data.bin 100 10000 1000 100

Ebenfalls enthält der Ordner die coordinate.cpp/h Dateien. In diesen Dateien ist die Schnittstelle zum beschreiben der .bin-Dateien mit Messwerten implementiert.
Als Orientierung soll testbench.cpp dienen. Hier könnt ihr nachschauen wie coordinate.cpp/h umgesetzt wurde.
(Nicht kompilieren weil die 'nbody.h' Datei fehlt, da ihr die Simulation selber implementieren sollt).

Wichtiges aus coordinate.cpp/h:
	#include "coordinate.h" -- ist klar!
	
	coordinate(const char* filename, int circles, int timesteps, int maxmass, int minmass, COORDINATE_SIZE cSize);
	*COORDINATE_SIZE ist ein struct welches die Achsenlänge für die Visualisierung festlegt. Einfach bei 400 belassen (siehe testbench.cpp Zeile 11-13 und Zeile 68).

	klassenname.initMass(int circleIdx, double bodymass);	(siehe testbench.cpp Zeile 86) 
	Diese Funktion ist wichtig um die Masse der einzelnen NBodys in die .bin-Datei zu schreiben. Die Visualisierung wird diese Wert lesen und die Kugeln in der Graphik entsprechend groß machen.
	WICHTIG! Davor sollten die Massen für die NBody bereits definiert sein!!! In der testbench.cpp wird das in Zeile 77 beim Erstellen der Klasseninstanz nbody3d simulation(...) gemacht.

	int writeTimestep(NBODY* cords, int maxcircles); diese Funktion schreibt die Position der NBody in die Datei, als komplette Zeile: "nbody0,x,y,z:nbody1,x,y,z:nbody2,x,y,z:..." als Beispiel.

Und mehr braucht ihr eigentlich nicht.

nbody_testdata Ordner: enthält bereits erstellte Messdaten. Ihr könnt diese in euren Home-Ordner kopieren und mit dem nbody_visualisation Programm anschauen.
 
