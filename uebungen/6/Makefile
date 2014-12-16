heat:
	mpic++ -o heat heat.cpp

clean:
	rm -rf *.o heat

run_single:
	mpirun -host localhost -np 2 ./heat
