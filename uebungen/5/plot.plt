set term eps enhanced color
set output "plot.eps"

set title "Heat Realaxation FLOPS"

set logscale x 2
set logscale y

set xtics nomirror
set ytics nomirror
set y2tics

set xrange [100:6000]
set yrange [0.01:1000]
set y2range [0.9:2.1]

set xlabel "Grid size"
set ylabel "Time in [ms]"
set y2label "GFLOP/s"

plot "time.dat" u 1:4 w linespoints pt 7 ps 0.5 axis x1y2 notitle,\
     "time.dat" u 1:($2)*1000 w linespoints pt 7 ps 0.5 axis x1y1 notitle 
