3_3:
	mpic++ -o pp pingpong.cpp

clean:
	rm -rf *.o pp

run_single:
	mpirun -host creek04,creek04 -np 2 ./pp

run_double:
	mpirun -host creek04,creek06 -np 2 ./pp
