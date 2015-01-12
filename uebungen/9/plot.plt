set logscale xy
set xlabel 'problem size in Bytes'
set ylabel 'time in 1000ns'
plot 'pinned.dat' u 1:2 with lines title 'serial time',  'pinned.dat' u 1:3 with lines title 'parallel time ',  'pinned.dat' u 1:4 with lines title 'data movements pinned','pageable.dat' u 1:4 with lines title 'data movements pageable'
