#!/usr/bin/env make

FILE=simpleCUDA

all: $(FILE).cu
	nvcc $(FILE).cu -o $(FILE) --ptxas-options=-v

clean:
	\rm -f $(FILE).linkinfo $(FILE)

