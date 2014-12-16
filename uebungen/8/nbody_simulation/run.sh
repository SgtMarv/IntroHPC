#run script
echo 'exec: 50 1000 10'
echo ' '
./main testdata_50_1000_100.bin 50 10000 1000 10
echo 'exec: 75 1500 100'
echo ' '
./main testdata_75_1500_100.bin 75 10000 1500 100
echo 'exec: 100 5000 1000'
echo ' '
./main testdata_100_5000_1000.bin 100 10000 5000 1000
echo 'exec: 125 25000 10000'
echo ' '
./main testdata_125_25000_1000.bin 125 10000 25000 10000
echo 'exec: 150 100000 1'
echo ' '
./main testdata_150_10000_1.bin 150 10000 100000 1
