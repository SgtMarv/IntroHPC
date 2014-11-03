reset;
set xlabel "number of processes"
set ylabel "avarage time of one barrier (1000 total)"
plot "./data.dat" u 1:3 w lp title "our own centrel barrier", "./data.dat" u 1:5 w lp title "MPI Barrier";
