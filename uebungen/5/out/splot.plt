set terminal jpeg
set output sprintf("state%s.jpeg",out)
set size ratio 1.0
set view map
set cbrange [0:127]
set title sprintf("State: %d",in)
unset key

stats sprintf("./state%d.dat",in) u 1:2 nooutput
set xrange [-0.5:STATS_max_x+0.5]
set yrange [-0.5:STATS_max_y+0.5]
     
splot sprintf("./state%d.dat",in) using 2:1:3 with points pt 5 ps 1 palette 
