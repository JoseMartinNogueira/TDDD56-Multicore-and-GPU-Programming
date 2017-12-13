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

#define KERNELSIZE 2

#define MAXBLOCKDIM 32

//https://www.evl.uic.edu/sjames/cs525/final.html


int searchBlockSize(const int imageSize, const int maxBlockSize){
    
    for(int i = maxBlockSize; i>0; i--)
    {
        if(imageSize%i==0)
            return i;
    }
    return 1;
}

__global__ void filter_shared(unsigned char *image, unsigned char *out, const unsigned int imagesizey, const unsigned int imagesizex)
{ 
	 
  int x = blockIdx.x * blockDim.x + threadIdx.x - blockIdx.x * KERNELSIZE * 2;
  int y = blockIdx.y * blockDim.y + threadIdx.y - blockIdx.y * KERNELSIZE * 2;

  int xid = threadIdx.x;
  int yid = threadIdx.y;

  int offsetImage = x + y * imagesizex;
  int offsetShared = xid + yid * blockDim.x;
  
  
  __shared__ unsigned char shared_img[MAXBLOCKDIM * MAXBLOCKDIM * 3];

  shared_img[offsetShared * 3 + 0] = image[offsetImage * 3 + 0];
  shared_img[offsetShared * 3 + 1] = image[offsetImage * 3 + 1];
  shared_img[offsetShared * 3 + 2] = image[offsetImage * 3 + 2];

  __syncthreads();
  
  
  bool blockInLeft = blockIdx.x == 0;
  bool blockInRight = blockIdx.x == gridDim.x - 1;
  bool blockInTop = blockIdx.y == 0;
  bool blockInBot = blockIdx.y == gridDim.y - 1;

  bool left = xid < KERNELSIZE;
  bool right = xid >= blockDim.x - KERNELSIZE;
  bool top = yid < KERNELSIZE;
  bool bottom = yid >= blockDim.y - KERNELSIZE;

  bool middleXarea = !left && !right;
  bool middleYarea = !top && !bottom;
  
  bool topLeft = blockInTop && blockInLeft && top && left;
  bool topEdge = blockInTop && top && middleXarea;
  bool topRight = blockInTop && blockInRight && top && right;
  
  bool leftEdge = blockInLeft && left && middleYarea;
  bool middle = middleXarea && middleYarea;
  bool rightEdge = blockInRight && right && middleYarea;
  
  bool botLeft = blockInBot && blockInLeft && bottom && left;
  bool botEdge = blockInBot && bottom && middleXarea;
  bool botRight = blockInBot && blockInRight && bottom && right;
  
  
  int i, j;
  unsigned int sumR, sumG, sumB;
  sumR=0;
  sumG=0;
  sumB=0;
  int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);
  
  
  if ( topLeft || topEdge || topRight ||
	  leftEdge || middle || rightEdge ||
	   botLeft || botEdge || botRight
	  )
  {
	  for (i=-KERNELSIZE;i<=KERNELSIZE;++i) {
		for (j=-KERNELSIZE;j<=KERNELSIZE;++j) {
			int xx = min(max(xid + j, 0), imagesizex - 1);
			int yy = min(max(yid + i, 0), imagesizey - 1);
			sumR += shared_img[(xx + yy * blockDim.x) * 3 + 0];
			sumG += shared_img[(xx + yy * blockDim.x) * 3 + 1];
			sumB += shared_img[(xx + yy * blockDim.x) * 3 + 2];
		}
	  }
	  out[offsetImage * 3 + 0] = sumR / divby;
	  out[offsetImage * 3 + 1] = sumG / divby;
	  out[offsetImage * 3 + 2] = sumB / divby;
  }

}

__global__ void filter_horizontal(unsigned char *image, unsigned char *out, const unsigned int imagesizey, const unsigned int imagesizex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x - blockIdx.x * KERNELSIZE * 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int xid = threadIdx.x;
    int yid = threadIdx.y;

    int offsetImage = x + y * imagesizex;
    int offsetShared = xid + yid * blockDim.x;


    __shared__ unsigned char shared_img[MAXBLOCKDIM * MAXBLOCKDIM * 3];

    shared_img[offsetShared * 3 + 0] = image[offsetImage * 3 + 0];
    shared_img[offsetShared * 3 + 1] = image[offsetImage * 3 + 1];
    shared_img[offsetShared * 3 + 2] = image[offsetImage * 3 + 2];

    __syncthreads();


    bool blockInLeft = blockIdx.x == 0;
    bool blockInRight = blockIdx.x == gridDim.x - 1;

    bool left = xid < KERNELSIZE;
    bool right = xid >= blockDim.x - KERNELSIZE;

    bool middleXarea = !left && !right;

    bool leftEdge = blockInLeft && left;
    bool middle = middleXarea;
    bool rightEdge = blockInRight && right;

    int j;
    unsigned int sumR, sumG, sumB;
    sumR=0;
    sumG=0;
    sumB=0;
    int divby = (2*KERNELSIZE+1);

    if (
        leftEdge || middle || rightEdge
        )
    {

          for (j=-KERNELSIZE;j<=KERNELSIZE;++j) {
              int xx = min(max(xid + j, 0), imagesizex - 1);
              int yy = yid;
              sumR += shared_img[(xx + yy * blockDim.x) * 3 + 0];
              sumG += shared_img[(xx + yy * blockDim.x) * 3 + 1];
              sumB += shared_img[(xx + yy * blockDim.x) * 3 + 2];
          }

        out[offsetImage * 3 + 0] = sumR / divby;
        out[offsetImage * 3 + 1] = sumG / divby;
        out[offsetImage * 3 + 2] = sumB / divby;
    }

}


__global__ void filter_vertical(unsigned char *image, unsigned char *out, const unsigned int imagesizey, const unsigned int imagesizex)
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y - blockIdx.y * KERNELSIZE * 2;

  int xid = threadIdx.x;
  int yid = threadIdx.y;

  int offsetImage = x + y * imagesizex;
  int offsetShared = xid + yid * blockDim.x;


  __shared__ unsigned char shared_img[MAXBLOCKDIM * MAXBLOCKDIM * 3];

  shared_img[offsetShared * 3 + 0] = image[offsetImage * 3 + 0];
  shared_img[offsetShared * 3 + 1] = image[offsetImage * 3 + 1];
  shared_img[offsetShared * 3 + 2] = image[offsetImage * 3 + 2];

  __syncthreads();


  bool blockInTop = blockIdx.y == 0;
  bool blockInBot = blockIdx.y == gridDim.y - 1;

  bool top = yid < KERNELSIZE;
  bool bottom = yid >= blockDim.y - KERNELSIZE;

  bool middleYarea = !top && !bottom;

  bool topEdge = blockInTop && top;
  bool middle = middleYarea;
  bool botEdge = blockInBot && bottom;



  int i;
  unsigned int sumR, sumG, sumB;
  sumR=0;
  sumG=0;
  sumB=0;
  int divby = (2*KERNELSIZE+1);


  if ( topEdge ||
      middle ||
       botEdge
      )
  {
      for (i=-KERNELSIZE;i<=KERNELSIZE;++i) {

            int xx = xid;
            int yy = min(max(yid + i, 0), imagesizey - 1);
            sumR += shared_img[(xx + yy * blockDim.x) * 3 + 0];
            sumG += shared_img[(xx + yy * blockDim.x) * 3 + 1];
            sumB += shared_img[(xx + yy * blockDim.x) * 3 + 2];

      }
      out[offsetImage * 3 + 0] = sumR / divby;
      out[offsetImage * 3 + 1] = sumG / divby;
      out[offsetImage * 3 + 2] = sumB / divby;
  }

}


__global__ void filter_GaussHorizontal(unsigned char *image, unsigned char *out, const unsigned int imagesizey, const unsigned int imagesizex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x - blockIdx.x * KERNELSIZE * 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int xid = threadIdx.x;
    int yid = threadIdx.y;

    int offsetImage = x + y * imagesizex;
    int offsetShared = xid + yid * blockDim.x;


    __shared__ unsigned char shared_img[MAXBLOCKDIM * MAXBLOCKDIM * 3];

    shared_img[offsetShared * 3 + 0] = image[offsetImage * 3 + 0];
    shared_img[offsetShared * 3 + 1] = image[offsetImage * 3 + 1];
    shared_img[offsetShared * 3 + 2] = image[offsetImage * 3 + 2];

    __syncthreads();


    bool blockInLeft = blockIdx.x == 0;
    bool blockInRight = blockIdx.x == gridDim.x - 1;

    bool left = xid < KERNELSIZE;
    bool right = xid >= blockDim.x - KERNELSIZE;

    bool middleXarea = !left && !right;

    bool leftEdge = blockInLeft && left;
    bool middle = middleXarea;
    bool rightEdge = blockInRight && right;

    int j;
    unsigned int sumR, sumG, sumB;
    sumR=0;
    sumG=0;
    sumB=0;
    int divby = 16;
    int gauss[5] = {1,4,6,4,1};

    if (
        leftEdge || middle || rightEdge
        )
    {

          for (j=-KERNELSIZE;j<=KERNELSIZE;++j) {
              int xx = min(max(xid + j, 0), imagesizex - 1);
              int yy = yid;
              sumR += shared_img[(xx + yy * blockDim.x) * 3 + 0]*gauss[KERNELSIZE+j];
              sumG += shared_img[(xx + yy * blockDim.x) * 3 + 1]*gauss[KERNELSIZE+j];
              sumB += shared_img[(xx + yy * blockDim.x) * 3 + 2]*gauss[KERNELSIZE+j];
          }

        out[offsetImage * 3 + 0] = sumR / divby;
        out[offsetImage * 3 + 1] = sumG / divby;
        out[offsetImage * 3 + 2] = sumB / divby;
    }

}


__global__ void filter_GaussVertical(unsigned char *image, unsigned char *out, const unsigned int imagesizey, const unsigned int imagesizex)
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y - blockIdx.y * KERNELSIZE * 2;

  int xid = threadIdx.x;
  int yid = threadIdx.y;

  int offsetImage = x + y * imagesizex;
  int offsetShared = xid + yid * blockDim.x;


  __shared__ unsigned char shared_img[MAXBLOCKDIM * MAXBLOCKDIM * 3];

  shared_img[offsetShared * 3 + 0] = image[offsetImage * 3 + 0];
  shared_img[offsetShared * 3 + 1] = image[offsetImage * 3 + 1];
  shared_img[offsetShared * 3 + 2] = image[offsetImage * 3 + 2];

  __syncthreads();


  bool blockInTop = blockIdx.y == 0;
  bool blockInBot = blockIdx.y == gridDim.y - 1;

  bool top = yid < KERNELSIZE;
  bool bottom = yid >= blockDim.y - KERNELSIZE;

  bool middleYarea = !top && !bottom;

  bool topEdge = blockInTop && top;
  bool middle = middleYarea;
  bool botEdge = blockInBot && bottom;



  int i;
  unsigned int sumR, sumG, sumB;
  sumR=0;
  sumG=0;
  sumB=0;
  int divby = 16;
  int gauss[5] = {1,4,6,4,1};


  if ( topEdge ||
      middle ||
       botEdge
      )
  {
      for (i=-KERNELSIZE;i<=KERNELSIZE;++i) {

            int xx = xid;
            int yy = min(max(yid + i, 0), imagesizey - 1);
            sumR += shared_img[(xx + yy * blockDim.x) * 3 + 0]*gauss[KERNELSIZE+i];
            sumG += shared_img[(xx + yy * blockDim.x) * 3 + 1]*gauss[KERNELSIZE+i];
            sumB += shared_img[(xx + yy * blockDim.x) * 3 + 2]*gauss[KERNELSIZE+i];

      }
      out[offsetImage * 3 + 0] = sumR / divby;
      out[offsetImage * 3 + 1] = sumG / divby;
      out[offsetImage * 3 + 2] = sumB / divby;
  }

}
	
__global__ void filter_median(unsigned char *image, unsigned char *out, const unsigned int imagesizey, const unsigned int imagesizex)
{

  int x = blockIdx.x * blockDim.x + threadIdx.x - blockIdx.x * KERNELSIZE * 2;
  int y = blockIdx.y * blockDim.y + threadIdx.y - blockIdx.y * KERNELSIZE * 2;

  int xid = threadIdx.x;
  int yid = threadIdx.y;

  int offsetImage = x + y * imagesizex;
  int offsetShared = xid + yid * blockDim.x;


  __shared__ unsigned char shared_img[MAXBLOCKDIM * MAXBLOCKDIM * 3];

  shared_img[offsetShared * 3 + 0] = image[offsetImage * 3 + 0];
  shared_img[offsetShared * 3 + 1] = image[offsetImage * 3 + 1];
  shared_img[offsetShared * 3 + 2] = image[offsetImage * 3 + 2];

  __syncthreads();


  bool blockInLeft = blockIdx.x == 0;
  bool blockInRight = blockIdx.x == gridDim.x - 1;
  bool blockInTop = blockIdx.y == 0;
  bool blockInBot = blockIdx.y == gridDim.y - 1;

  bool left = xid < KERNELSIZE;
  bool right = xid >= blockDim.x - KERNELSIZE;
  bool top = yid < KERNELSIZE;
  bool bottom = yid >= blockDim.y - KERNELSIZE;

  bool middleXarea = !left && !right;
  bool middleYarea = !top && !bottom;

  bool topLeft = blockInTop && blockInLeft && top && left;
  bool topEdge = blockInTop && top && middleXarea;
  bool topRight = blockInTop && blockInRight && top && right;

  bool leftEdge = blockInLeft && left && middleYarea;
  bool middle = middleXarea && middleYarea;
  bool rightEdge = blockInRight && right && middleYarea;

  bool botLeft = blockInBot && blockInLeft && bottom && left;
  bool botEdge = blockInBot && bottom && middleXarea;
  bool botRight = blockInBot && blockInRight && bottom && right;


  int i, j;
  unsigned int sumR, sumG, sumB;
  sumR=0;
  sumG=0;
  sumB=0;
  int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);

  unsigned int red[256];
  unsigned int green[256];
  unsigned int blue[256];
  memset(red, 0, 256*sizeof(unsigned int));
  memset(green, 0, 256*sizeof(unsigned int));
  memset(blue, 0, 256*sizeof(unsigned int));

  if ( topLeft || topEdge || topRight ||
      leftEdge || middle || rightEdge ||
       botLeft || botEdge || botRight
      )
  {
      for (i=-KERNELSIZE;i<=KERNELSIZE;++i) {
        for (j=-KERNELSIZE;j<=KERNELSIZE;++j) {
            int xx = min(max(xid + j, 0), imagesizex - 1);
            int yy = min(max(yid + i, 0), imagesizey - 1);
            ++red[shared_img[(xx + yy * blockDim.x) * 3 + 0]];
            ++green[shared_img[(xx + yy * blockDim.x) * 3 + 1]];
            ++blue[shared_img[(xx + yy * blockDim.x) * 3 + 2]];
        }
      }
      int r = -1, g = -1, b = -1;
      while (sumR < (2 * KERNELSIZE + 1) / 2)
      {
          sumR += red[(++r)];
      }
      while (sumG < (2 * KERNELSIZE + 1) / 2)
      {
          sumG += green[(++g)];
      }
      while (sumB < (2 * KERNELSIZE + 1) / 2)
      {
          sumB += blue[(++b)];
      }
      out[offsetImage * 3 + 0] = r;
      out[offsetImage * 3 + 1] = g;
      out[offsetImage * 3 + 2] = b;
  }

}




// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages_shared()
{
	//if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	//{
		//printf("Kernel size out of bounds!\n");
		//return;
	//}

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
	
    int blockX=searchBlockSize(imagesizex - 2 * KERNELSIZE, MAXBLOCKDIM - KERNELSIZE * 2) + KERNELSIZE * 2;
    int blockY=searchBlockSize(imagesizey - 2 * KERNELSIZE, MAXBLOCKDIM - KERNELSIZE * 2) + KERNELSIZE * 2;
	printf("bx=%d by=%d\n",blockX,blockY);
	dim3 block(blockX,blockY);
	int gridX=(imagesizex - 2 * KERNELSIZE) / (blockX - KERNELSIZE * 2);
	int gridY=(imagesizey - 2 * KERNELSIZE) / (blockY - KERNELSIZE * 2);
	printf("gx=%d gy=%d\n",gridX,gridY);
	dim3 grid(gridX,gridY);	
	filter_shared<<<grid,block>>>(dev_input, dev_bitmap, imagesizey, imagesizex); // Awful load balance
	
	cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

void computeImages_separated()
{
    pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
    cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
    cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
    cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
    int blockX1=searchBlockSize(imagesizex - 2 * KERNELSIZE, MAXBLOCKDIM - KERNELSIZE * 2) + KERNELSIZE * 2;
    int blockY1=searchBlockSize(imagesizey, MAXBLOCKDIM);
    printf("bx1=%d by1=%d\n",blockX1,blockY1);
    dim3 block1(blockX1,blockY1);
    int gridX1=(imagesizex - 2 * KERNELSIZE) / (blockX1 - KERNELSIZE * 2);
    int gridY1=imagesizey / blockY1;
    printf("gx1=%d gy1=%d\n",gridX1,gridY1);
    dim3 grid1(gridX1,gridY1);

    int blockX2=searchBlockSize(imagesizex, MAXBLOCKDIM);
    int blockY2=searchBlockSize(imagesizey - 2 * KERNELSIZE, MAXBLOCKDIM - KERNELSIZE * 2) + KERNELSIZE * 2;
    printf("bx2=%d by2=%d\n",blockX2,blockY2);
    dim3 block2(blockX2,blockY2);
    int gridX2=imagesizex/blockX2;
    int gridY2=(imagesizey - 2 * KERNELSIZE) / (blockY2 - KERNELSIZE * 2);
    printf("gx2=%d gy2=%d\n",gridX2,gridY2);
    dim3 grid2(gridX2,gridY2);


    filter_horizontal<<<grid1,block1>>>(dev_input, dev_bitmap, imagesizey, imagesizex);
    cudaThreadSynchronize();
    filter_vertical<<<grid2,block2>>>(dev_bitmap, dev_input, imagesizey, imagesizex);

    cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
    cudaFree( dev_bitmap );
    cudaFree( dev_input );

}

void computeImages_Gauss()
{
    pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
    cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
    cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
    cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
    int blockX1=searchBlockSize(imagesizex - 2 * KERNELSIZE, MAXBLOCKDIM - KERNELSIZE * 2) + KERNELSIZE * 2;
    int blockY1=searchBlockSize(imagesizey, MAXBLOCKDIM);
    printf("bx1=%d by1=%d\n",blockX1,blockY1);
    dim3 block1(blockX1,blockY1);
    int gridX1=(imagesizex - 2 * KERNELSIZE) / (blockX1 - KERNELSIZE * 2);
    int gridY1=imagesizey / blockY1;
    printf("gx1=%d gy1=%d\n",gridX1,gridY1);
    dim3 grid1(gridX1,gridY1);

    int blockX2=searchBlockSize(imagesizex, MAXBLOCKDIM);
    int blockY2=searchBlockSize(imagesizey - 2 * KERNELSIZE, MAXBLOCKDIM - KERNELSIZE * 2) + KERNELSIZE * 2;
    printf("bx2=%d by2=%d\n",blockX2,blockY2);
    dim3 block2(blockX2,blockY2);
    int gridX2=imagesizex/blockX2;
    int gridY2=(imagesizey - 2 * KERNELSIZE) / (blockY2 - KERNELSIZE * 2);
    printf("gx2=%d gy2=%d\n",gridX2,gridY2);
    dim3 grid2(gridX2,gridY2);


    filter_GaussHorizontal<<<grid1,block1>>>(dev_input, dev_bitmap, imagesizey, imagesizex);
    cudaThreadSynchronize();
    filter_GaussVertical<<<grid2,block2>>>(dev_bitmap, dev_input, imagesizey, imagesizex);

    cudaThreadSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
    cudaFree( dev_bitmap );
    cudaFree( dev_input );

}

void computeImages_median()
{
    //if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
    //{
        //printf("Kernel size out of bounds!\n");
        //return;
    //}

    pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
    cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
    cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
    cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);

    int blockX=searchBlockSize(imagesizex - 2 * KERNELSIZE, MAXBLOCKDIM - KERNELSIZE * 2) + KERNELSIZE * 2;
    int blockY=searchBlockSize(imagesizey - 2 * KERNELSIZE, MAXBLOCKDIM - KERNELSIZE * 2) + KERNELSIZE * 2;
    printf("bx=%d by=%d\n",blockX,blockY);
    dim3 block(blockX,blockY);
    int gridX=(imagesizex - 2 * KERNELSIZE) / (blockX - KERNELSIZE * 2);
    int gridY=(imagesizey - 2 * KERNELSIZE) / (blockY - KERNELSIZE * 2);
    printf("gx=%d gy=%d\n",gridX,gridY);
    dim3 grid(gridX,gridY);
    filter_median<<<grid,block>>>(dev_input, dev_bitmap, imagesizey, imagesizex); // Awful load balance

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
        image = readppm((char *)"maskros-noisy.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();
	
	
    computeImages_median();
	
	double time = 1000*GetSeconds();
	printf("%f  ms\n", time );

// You can save the result to a file like this:
    writeppm("noisyMedian.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}
