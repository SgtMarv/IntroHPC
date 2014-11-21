5: 
	g++ -o heat heat.cpp

clean:
	rm -rf *.o heat

run:
	./heat $(n) $(it) 0

gif:
	./heat $(n) $(it) 1 && cd out && ./gif.sh $(it) && cd ..


