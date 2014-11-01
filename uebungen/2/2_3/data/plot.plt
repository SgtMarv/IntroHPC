set term eps enhanced color
set output "matmult.eps"

set title "Execution time for matrix multiplication"


set ytics nomirror
set xtics nomirror
set y2tics 

set grid y


set xlabel "Run No."
set ylabel "Time [s]"
set y2label "GLOPS"

set yrange [332.5: 334]
set y2range [0.0514: 0.0517]

mean = 333.19
stddev = 0.21

set label 1 gprintf("Mean = %.2fs", mean) at 3.755, 332.75
set label 2 gprintf("StdDev = %.2fs", stddev) at 3.5, 332.65

plot mean+stddev with filledcurves y1=mean lc rgb "#bbbbdd" notitle,\
     mean-stddev with filledcurves y1=mean lc rgb "#bbbbdd" notitle,\
     "time.dat" using ($1) pt 7 ps 0.5 lc rgb "red" title "Time per MatMult",\
     "time.dat" using ((2*2048**3)/($1))/(10**9) axis x1y2 pt 7 ps 0.5 lc rgb "blue" title "FLOPS",\
     mean lc rgb "blue"
