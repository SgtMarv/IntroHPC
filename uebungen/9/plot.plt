set logscale xy
set xlabel 'problem size in B'
set ylabel 'time'
plot 'pinned.dat' u 1:2 with lines title 'serial time',  'pinned.dat' u 1:3 with lines title 'parallel time ',  'pinned.dat' u 1:4 with lines title 'data movements'
