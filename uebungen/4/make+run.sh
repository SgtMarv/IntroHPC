mpic++ matrix_multiply.cc -o mamu

mpirun -host creek05 -np 3 ./mamu 
