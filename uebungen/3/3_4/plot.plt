set logscale x
set logscale y
set grid
set key left
set ylabel 'bandwith in B/s'
set xlabel 'messege size in KB'
plot '1node.dat' u 2:3 with lp title '1node blocking', '2nodes.dat' u 2:3 with lp  title '2nodes blocking', '2nodes.dat' u 2:4 w lp title '2nodes non blocking', '1node.dat' u 2:4 w lp title '1node non blocking'
