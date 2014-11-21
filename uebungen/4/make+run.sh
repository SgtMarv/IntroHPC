mpic++ matrix_multiply2.cc -o mamu

mpirun -host creek01,creek02,creek03,creek04 -np 16 ./mamu  
