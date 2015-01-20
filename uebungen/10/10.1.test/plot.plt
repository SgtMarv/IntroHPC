set xlabel 'problemsize'
set ylabel 'time in s'
set logscale xy
plot 'data.dat' u 1:2 w lp title 'cpu', 'data.dat' u 1:3 w lp title 'gpu' ,'data.dat' u 1:4 w lp title 'gpu+sendtime'