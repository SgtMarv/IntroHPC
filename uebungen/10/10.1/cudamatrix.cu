/*
 *
 * nullKernelAsync.cu
 *
 * Microbenchmark for throughput of asynchronous kernel launch.
 *
 * Build with: nvcc -I ../chLib <options> nullKernelAsync.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in 
 *    the documentation and/or other materials provided with the 
 *    distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>

#include "chTimer.h"

__global__void Matrixmulti(float A[N][N], float B[N][N], float C[N][N])
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y; 
	
	
	if(i<N&&j<N){for (int k=0;k<N;k++)
	{
	 C[i][j]+=A[i][k]*B[k][j]
	}}

}

int
main( int argc, char *argv[] )
{ 
	int matrixsize N=16;
	
	void *dmemA = cudaMalloc ( N*N*sizeof ( float ) ); // Allocate GPU memory
	void *dmemB = cudaMalloc ( N*N*sizeof ( float ) ); // Allocate GPU memory
	void *dmemC = cudaMalloc ( N*N*sizeof ( float ) ); // Allocate GPU memory
	
    void *hmemA = malloc ( N*N*sizeof ( float ) ); // Allocate CPU memory
	void *hmemB = malloc ( N*N*sizeof ( float ) ); // Allocate CPU memory
	void *hmemC = malloc ( N*N*sizeof ( float ) ); // Allocate CPU memory
	
	
	
	
	cudaMemcpy ( dmemA, hmemA, N*N*sizeof ( float ), cudaMemcpyHostToDevice ); 
	cudaMemcpy ( dmemB, hmemB, N*N*sizeof ( float ), cudaMemcpyHostToDevice ); 
	
    printf( "Measuring... " ); fflush( stdout );

    chTimerTimestamp start, stop;

	dim3 dimBock(N,N);
	
	cudamemcopyy 
	
    chTimerGetTime( &start );
    
        matrixmulti<<<1,dimBlock>>>(dmemA,dmemB,dmemC,N);
    
    cudaThreadSynchronize();
    chTimerGetTime( &stop );

    {
        double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (float) cIterations;

        printf( "%.2f us\n", usPerLaunch );
    }
	
	cudaMemcpy ( hmemC, dmemC, N*N*sizeof ( float ), cudaMemcpyDeviceToHost ); 
	
	
	cudafree(dmemA);
	cudafree(dmemB);
	cudafree(dmemC);
	free(hmemA);
	free(hmemB);
	free(hmemC);
    return 0;
}
