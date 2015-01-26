set term eps enhanced color
set output "latency.eps"

set title "MPI Latency Ethernet vs. Extoll1"

set logscale x 2

set xtics nomirror
set ytics nomirror
set y2tics

set xrange [0.5:1500]
set yrange [0:18]
set y2range [0:18]

set xlabel "Message size in [kB]"
set ylabel "Time in [ms]"
set y2label "Time in [ms]"


plot "ethernet_lat_avg.dat" u 1:($2)*1000:($3)*1000 w yerrorbars axis x1y1 pt 7 ps 0.5 lc rgb "red" title "Ethernet"
