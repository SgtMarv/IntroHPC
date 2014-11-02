set term eps enhanced color
set output "opt.eps"

set title "Execution time for matrix multiplication - optimized"


set ytics nomirror
set xtics nomirror
set y2tics 

set grid y

set xlabel "Run No."
set ylabel "Time [s]"
set y2label "GLOPS"

set yrange [39.0: 41.0]
set y2range [0.42: 0.44]

mean = 39.95
stddev = 0.17

set label 1 gprintf("Mean = %.2fs", mean) at 3.755, 39.35
set label 2 gprintf("StdDev = %.2fs", stddev) at 3.5, 39.2

plot mean+stddev with filledcurves y1=mean lc rgb "#bbbbdd" notitle axis x1y1,\
     mean-stddev with filledcurves y1=mean lc rgb "#bbbbdd" notitle axis x1y1,\
     "opt.dat" using ($1) pt 7 ps 0.5 lc rgb "red" title "Time per MatMultOpt" axis x1y1,\
     "opt.dat" using ((2*2048**3)/($1))/(10**9) axis x1y2 pt 7 ps 0.5 lc rgb "blue" title "FLOPS",\
     mean lc rgb "blue"
