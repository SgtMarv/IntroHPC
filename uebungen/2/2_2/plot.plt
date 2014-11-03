reset;

set ylabel "avarage time of one barrier (100 total)"
plot "./data.dat" u 1:3 w lp title "our own centrel barrier", "./data.dat" u 1:5 w lp title "MPI Barrier";
set terminal eps
set output "bild.eps"
set xlabel "number of processes"
replot;