#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <string>
#include <math.h>
#include <assert.h>

#define BLOCK_SIZE 16

using namespace std;


__global__ void Convolution(float* A, float* B, float* C, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float sum;
   
    //A * B = C
    if(row<numCRows&&col<numCCols) {
        sum =0;

		// Part 1. write GPU kernel code for executing convolution
		// **convolution with the entire filter becomes the one element of the matrix C
		// ***row/col indicates the index of output matrix C
		for(size_t i = 0; i < numBCols; i++) {
			for(size_t j = 0; j < numBRows; j++) {
				sum += A[row*numACols + i*numACols + col + j] * B[i*numBCols + j];
			}
		}

		C[row*numCCols + col] = sum; 
		//Part 1 ends
    }
}

__host__ void cpu_Convolution(float *A, float *B, float *C, int asize, int bsize){

	int csize=asize-bsize+1;
	int sum;
	for(int i=0;i<csize;i++){
		
		for (int j=0;j<csize;j++){
			sum=0;
			for(int k=0;k<bsize;k++){
				
				for(int l=0;l<bsize;l++){
					sum+=A[i*asize+k*asize+j+l]*B[k*bsize+l];
				}
			}
			C[i*csize+j]=sum;
		}
	}
}

void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() %10;
}

int main()
{
	srand(2006);
	int a, b, c;
	cudaEvent_t start_G, stop_G;
	float gpu_miliseconds, cpu_miliseconds;
	cudaEventCreate(&start_G);
	cudaEventCreate(&stop_G);
	printf("Please type in the size of input and filter ( type in '5 3'-> 5 x 5 matrix and 3 x 3 filter) \n");
    scanf("%d %d", &a, &b); // determine the size of input A and filter B
	
	/*Part 2-1.	initilize matrix and filter
	when the input and filter size are a and b, what is the output size c?
	c= ??; 

	Part 2-1 ends
	*/
	c = a - b + 1;

	unsigned int size_A = a * a;
	unsigned int size_B = b * b;
	unsigned int size_C = c * c;

	/*Part 2-2. initialize matrix and filter
	allocate memory in host DRAM
	float *h_A, *h_B, *h_C, *h_C_cpu;
	*/
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*)malloc(mem_size_A);

	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*)malloc(mem_size_B);

	unsigned int mem_size_C = sizeof(float) * size_C;
	float* h_C = (float*)malloc(mem_size_C);
	float* h_C_cpu = (float*)malloc(mem_size_C);

	randomInit(h_A, size_A);
	for (int i = 0; i < size_B; ++i)
    {
        h_B[i] = 1; // rand() %4;
    }

	/*Part 3. allocate device memory and memory copy from the host
	allocate memory space on GPU, then copy matrix A and B from the host to GPU
	float *d_A, *d_B, *d_C;		//h_A, h_B -> d_A, d_B

	Part 3 ends
	*/
	float* d_A;
	float* d_B;
	float* d_C;

	cudaMalloc((void**)&d_A, mem_size_A);
	cudaMalloc((void**)&d_B, mem_size_B);
	cudaMalloc((void**)&d_C, mem_size_C);

	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	unsigned int grid_rows= (c+BLOCK_SIZE-1) / BLOCK_SIZE;
	unsigned int grid_cols= (c+BLOCK_SIZE-1) / BLOCK_SIZE;

	dim3 dimGrid(grid_cols,grid_rows);	
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	
	cudaEventRecord(start_G,0);

	/*Part 4-1. launch GPU kerenl
	launch GPU kernel and execute convolution between A and B, providing values for C
	Part 4-1 ends
	*/
	Convolution<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, a, a, b, b, c, c);
	
	cudaDeviceSynchronize();

	

	cudaEventRecord(stop_G,0);
	cudaEventSynchronize(stop_G);

	/*Part 4-2. data copy from device to host
	transfer the calculated data from gpu to host ( d_C -> h_C)
	*/
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
	
	cudaEventElapsedTime(&gpu_miliseconds, start_G, stop_G);

	printf("\nTime took to compute matrix A(%d x %d) with filter B(%d x %d) on GPU is %f ms  \n \n", a, a,b,b, gpu_miliseconds);
    /* 디버깅용 결과 확인 시 주석 해제
	printf("matrix A\n");
	for (int i = 0;i < a;i++)
	{
		for (int j = 0;j < a;j++)
		{
			printf("%f\t", h_A[i*a + j]);
		}
		printf("\n");
	}printf("\n");
    printf("matrix B\n");
    for (int i = 0;i < b;i++)
	{
		for (int j = 0;j < b;j++)
		{
			printf("%f\t", h_B[i*b + j]);
		}
		printf("\n");
	}printf("\n");
    printf("matrix C\n");
	for (int i = 0;i < c;i++)
	{
		for (int j = 0;j < c;j++)
		{
			printf("%f\t", h_C[i*c + j]);
		}
		printf("\n");
	}
	*/
	cudaEventRecord(start_G, 0);
	cpu_Convolution(h_A, h_B, h_C_cpu, a,b);	// convolution execution with CPU
	cudaEventRecord(stop_G,0);
	cudaEventSynchronize(stop_G);
	cudaEventElapsedTime(&cpu_miliseconds, start_G, stop_G);
	printf("\nTime took to compute matrix A(%d x %d) with filter B(%d x %x) on CPU is %f ms  \n \n", a, a,b,b, cpu_miliseconds);
    /* 디버깅용 결과 확인 시 주석 해제
	for (int i = 0;i < c;i++)
	{
		for (int j = 0;j < c;j++)
		{
			printf("%f\t", h_C_cpu[i*c + j]);
		}
		printf("\n");
	}
	*/
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_cpu);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return EXIT_SUCCESS;
}

