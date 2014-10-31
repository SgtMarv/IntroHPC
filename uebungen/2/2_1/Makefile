2_1:
	mpic++ -o 2_1 2_1.cpp

clean:
	rm -rf *.o 2_1

run:
	mpirun -host creek01,creek02,creek03,creek04,creek05,creek06,creek07,creek08 -np $(proc) ./2_1 $(msg) $(v) 

run_opt: 
	mpirun -host creek04,creek04,creek04,creek04,creek04,creek04,creek04,creek04,creek05,creek05,creek05,creek05 -np $(proc) ./2_1 $(msg) $(v) 

