// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -o filter.o -arch=sm_30
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut  -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10

#define TILE_LENGTH 2

#define KSX 2
#define KSY 2

//https://www.evl.uic.edu/sjames/cs525/final.html

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizey, const unsigned int imagesizex, const int kernelsizex, const int kernelsizey)
{ 
	
  
	__shared__ float dataShared[(TILE_LENGTH+2*KSX)*3][TILE_LENGTH+2*KSY];
  
  //const int myoffset = threadIdx.x + 
						//blockIdx.x * blockDim.x + 
						//threadIdx.y* imagesizex + 
						//blockIdx.y*blockDim.y * imagesizex;
	int x,y;
	for(int aux=0; aux<3; aux++){
	
	int idx = blockIdx.x * blockDim.x*3 +aux + threadIdx.x*3;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idOffset = idx + idy * imagesizex*3;   
	
	
	//upper left
	x= idx - kernelsizex;
	y= idy - kernelsizey;
	if(x<0 || y<0)
		dataShared[threadIdx.x*3+aux][threadIdx.y] = 0;
	else
		dataShared[threadIdx.x*3+aux][threadIdx.y] = 
			image[idOffset - (kernelsizex*3+aux) - imagesizex*3 * kernelsizey];
			
	//upper right
	x= idx + kernelsizex;
	y= idy - kernelsizey;
	if(x>imagesizex-1 || y<0)
		dataShared[(threadIdx.x+blockDim.x)*3+aux][threadIdx.y] = 0;
	else
		dataShared[(threadIdx.x+blockDim.x)*3+aux][threadIdx.y] = 
			image[idOffset + (kernelsizex*3+aux) - imagesizex*3 * kernelsizey];
			
	//lower left
	x= idx - kernelsizex;
	y= idy + kernelsizey;
	if(x<0 || y>imagesizey-1)
		dataShared[(threadIdx.x)*3+aux][threadIdx.y+blockDim.y] = 0;
	else
		dataShared[(threadIdx.x)*3+aux][threadIdx.y+blockDim.y] = 
			image[idOffset - (kernelsizex*3+aux) + imagesizex*3 * kernelsizey];
			
	//lower right
	x= idx + kernelsizex;
	y= idy + kernelsizey;
	if(x>imagesizex-1 || y>imagesizey-1)
		dataShared[(threadIdx.x+blockDim.x)*3+aux][threadIdx.y+blockDim.y] = 0;
	else
		dataShared[(threadIdx.x+blockDim.x)*3+aux][threadIdx.y+blockDim.y] = 
			image[idOffset + (kernelsizex*3+aux) + imagesizex *3* kernelsizey];
	}
			
	__syncthreads();
	
	//int offSet = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y) *imagesizex;
					
	//x = kernelsizex*3 + threadIdx.x;
	//y = kernelsizey + threadIdx.y;
	//unsigned int sumx, sumy, sumz;
	//sumx=0;sumy=0;sumz=0;
	//int divby=(2*kernelsizex+1)*(2*kernelsizey+1);
	//for(int i = -kernelsizex; i <= kernelsizex; i++)
		//for(int j = -kernelsizey; j <= kernelsizey; j++)	
		//{	
			//sumx += dataShared[x+i*3][y+j];
			//sumy += dataShared[x+i*3+1][y+j];
			//sumz += dataShared[x+i*3+2][y+j];
		//}
	
	//out[offSet*3+0] = sumx/divby;
	//out[offSet*3+1] = sumy/divby;
	//out[offSet*3+2] = sumz/divby;
}
	
	

// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
	dim3 grid(imagesizex,imagesizey);
	filter<<<grid,1>>>(dev_input, dev_bitmap, imagesizey, imagesizex, kernelsizex, kernelsizey); // Awful load balance
	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"sol.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(KSX, KSY);
	
	double time = 1000*GetSeconds();
	printf("%f  ms\n", time );

// You can save the result to a file like this:
	writeppm("cosa.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}
