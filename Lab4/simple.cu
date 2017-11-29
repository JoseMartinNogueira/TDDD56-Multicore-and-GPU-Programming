// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

#include <math.h>

const int N = 16; 
const int blocksize = 16; 

__global__ 
void simple(float *n, float *c) 
{
	c[threadIdx.x] = sqrt(n[threadIdx.x]);
}

/*__global__ 
void simple(float *c) 
{
	c[threadIdx.x] = threadIdx.x;
}*/

int main()
{
	float *c = new float[N];	
	float *cd;
	const int size = N*sizeof(float);
	float *src = new float[N];
	float *dst;

	for(int i=0; i<N; i++)
		src[i]=i*i*i*i;

	cudaMalloc( (void**)&cd, size );	
	cudaMalloc( (void**)&dst, size );
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice );
	simple<<<dimGrid, dimBlock>>>(dst,cd);
	//simple<<<dimGrid, dimBlock>>>(cd);
	cudaThreadSynchronize();
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cd );
	cudaFree( dst);
	
	for (int i = 0; i < N; i++)
		printf("%f\n ", c[i]);
	printf("\n");
	delete[] c;
	delete[] src;
	printf("done\n");
	return EXIT_SUCCESS;
}
