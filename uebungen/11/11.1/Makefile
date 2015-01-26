3_3:
	mpic++ -o latency latency.cpp

clean:
	rm -rf *.o latency

run:
	mpirun -host creek04,creek06 -np 2 ./latency
