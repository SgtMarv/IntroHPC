set term eps enhanced color
set output "runs.eps"

set title "Testing for stable number of messages m, using 12 processes"

set logscale x
set logscale y

set ytics nomirror
set xtics nomirror
set y2tics

set grid y2

set xrange [0.5: 15000]
set yrange [1e-6*1000:0.005*1000]
set y2range [2:20]

set xlabel "Nr. of Messages"
set ylabel "Time [ms]"
set y2label "Error [%]"

plot "runs.dat" using 1:($2)*1000:($3)*1000 with yerrorbars axes x1y1 pt 7 ps 0.3 lc rgb "blue" title "Average Time per Msg",\
     "runs.dat" using 1:($3)/($2)*100 with linespoints pt 4 ps 0.5 lc rgb "red" axes x1y2 title "Relative Error"
