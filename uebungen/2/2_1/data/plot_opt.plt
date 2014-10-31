set term eps enhanced color
set output "opt.eps"

set title "Measurement of optimized mapping for 12 processes and 10 messages"


set ytics nomirror
set xtics nomirror

set grid y


set xlabel "Run No."
set ylabel "Time [ms]"

set yrange [0.02:0.045]

mean = 2.9209948999999987e-05*1000
stddev = 2.641558754644956e-06*1000

set label 1 gprintf("Mean = %.4f", mean) at 37.45, 0.0235
set label 2 gprintf("StdDev = %.4f", stddev) at 35, 0.022

plot mean+stddev with filledcurves y1=mean lc rgb "#bbbbdd" notitle,\
     mean-stddev with filledcurves y1=mean lc rgb "#bbbbdd" notitle,\
     "opt_d.dat" using ($1)*1000 pt 7 ps 0.5 lc rgb "red" title "Average Time per Msg",\
     mean lc rgb "blue"

