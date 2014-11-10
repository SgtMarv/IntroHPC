

mpic++ -o flood_test flood_test.cc

rm 2nodes0.dat
rm 1node0.dat

mpirun -host creek06 -np 2 ./flood_test 15 100 >>1node0.dat
mpirun -host creek06,creek07 -np 2 ./flood_test 15 100 >>2nodes0.dat
