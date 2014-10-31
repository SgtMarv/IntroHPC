set term eps enhanced color
set output "time.eps"

set title "Average execution time for different number of processes"


set ytics nomirror
set xtics nomirror

set grid y

set xrange [0:26]
set yrange [0.04:0.1]

set xlabel "Nr. of Processes"
set ylabel "Time [ms]"

plot "time.dat" using 1:($2)*1000:($3)*1000 with errorbars pt 7 ps 0.3 lc rgb "blue" title "Average Time per Msg"
