#! /bin/bash

for i in *.jpeg; do rm $i; done
for i in *.gif; do rm $i; done

it=$1

if [ "$it" -gt 9 ];then 
    for i in $(seq 0 9); do gnuplot -e "in=$i; out='00$i'" splot.plt; done;
    if [ "$it" -gt 99 ];then
        for i in $(seq 10 99); do gnuplot -e "in=$i; out='0$i'" splot.plt; done;
        for i in $(seq 100 $it); do gnuplot -e "in=$i; out='$i'" splot.plt; done;
    else
        for i in $(seq 10 $it); do gnuplot -e "in=$i; out='0$i'" splot.plt; done;
    fi
else
    for i in $(seq 0 $it); do gnuplot -e "in=$i; out='00$i'" splot.plt; done;
fi

convert -delay 10 *.jpeg -loop 0 animation.gif

for i in *.jpeg; do rm $i; done
for i in *.dat; do rm $i; done
