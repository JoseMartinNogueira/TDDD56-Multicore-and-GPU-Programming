// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

#include <math.h>
#include "milli.h"

const int N = 16; 
const int blocksize = 16; 

__global__ 
void add_matrix(float *a, float *b, float *c) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int index = idy + idx * N;
	c[index]=a[index]+b[index];
}

/*__global__ 
void simple(float *c) 
{
	c[threadIdx.x] = threadIdx.x;
}*/

int main()
{
	int N2=N*N;
	int size = N2*sizeof(float);
	float *a=new float[N2];
	float *b=new float[N2];
	float *c=new float[N2];
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}

	float *aCuda;
	float *bCuda;
	float *cCuda;

	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);	

	cudaMalloc( (void**)&aCuda, size );	
	cudaMalloc( (void**)&bCuda, size );
	cudaMalloc( (void**)&cCuda, size );
	dim3 dimBlock( blocksize, blocksize );
	dim3 dimGrid( N/blocksize, N/blocksize );
	cudaMemcpy( aCuda, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( bCuda, b, size, cudaMemcpyHostToDevice );
	
	
	cudaEventRecord( startEvent, 0 );

	add_matrix<<<dimGrid, dimBlock>>>(aCuda,bCuda,cCuda);
	cudaThreadSynchronize();

	cudaEventRecord( stopEvent, 0 );

	cudaMemcpy( c, cCuda, size, cudaMemcpyDeviceToHost );

	cudaFree(aCuda);
	cudaFree(bCuda);
	cudaFree(cCuda); 


	for(int x=0; x<N; x++){
		for(int y=0; y<N; y++){
			printf("%0.2f  ", c[y*N+x]);
		}
		printf("\n");
	}

	cudaEventSynchronize( startEvent );
	cudaEventSynchronize( stopEvent );

	float time;
	cudaEventElapsedTime( &time, startEvent, stopEvent );

	printf("%0.8f\n", time);

	delete[] a;
	delete[] b;
	delete[] c;

	
	
	return EXIT_SUCCESS;
}
