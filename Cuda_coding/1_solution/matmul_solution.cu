#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16

/*************************************************
Function name: gpu_matrix_mult

Parameters:
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output pointer to a m X k matrix (C)

Note:
      grid and block should be configured as:
            dim3 dimGrid((k + BLOCKSIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
*************************************************/

__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k)
{
  //Part 1
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum;

  if(col < k && row < m)
  {
    sum = 0;
    for(int i = 0; i < n; i++)
    {
      sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
  }

}


/*************************************************
Function name: cpu_matrix_mult

Parameters:
            &a CPU host pointer to a n X n matrix (A)
            &b CPU host pointer to a n X n matrix (B)
            &c CPU host output pointer to a n X n matrix (C)
*************************************************/

__host__ void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

/*************************************************
Function name: main

Test and Compare
*************************************************/

int main(int argc, char const *argv[])
{
    int m, n, k;
    srand(time(0));
    
    printf("Please type in m n and k\n");
    scanf("%d %d %d", &m, &n, &k);

    //Part 2

    int *h_a, *h_b, *h_c, *h_cc;

    cudaMallocHost((void **) &h_a, sizeof(int) * m * n);
    cudaMallocHost((void **) &h_b, sizeof(int) * n * k);
    cudaMallocHost((void **) &h_c, sizeof(int) * m * k);
    cudaMallocHost((void **) &h_cc, sizeof(int) * m * k);
    //Part 2 ends
    
    // Random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    // Random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

    // Events to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start to measure the execution time of GPU version
    cudaEventRecord(start, 0);

    //Part 3
    int *d_a, *d_b, *d_c;

    cudaMalloc((void **) &d_a, sizeof(int) * m * n);
    cudaMalloc((void **) &d_b, sizeof(int) * n * k);
    cudaMalloc((void **) &d_c, sizeof(int) * m * k);

    cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);
    //Part 3 ends

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    //Part 4 
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);

    cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    //Part 4 ends

    cudaDeviceSynchronize();
    
    // Time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // GPU computing time elapse
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("\n\nGPU execution time on matrix multiplication of %dx%d . %dx%d: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);

    // CPU version
    cudaEventRecord(start, 0);

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("CPU execution time on matrix multiplication of %dx%d . %dx%d: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);

    // Validate the results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            // Uncomment below to see the actual results on both CPU and GPU
            // printf("CPU result [%d][%d]:%d == GPU result [%d][%d]:%d, ", i, j, h_cc[i * k + j], i, j, h_c[i * k + j]);
            if (h_cc[i * k + j] != h_c[i * k + j]) {
                all_ok = 0;
            }
        }
        // printf("\n");
    }

    // Compute the speedup
    if (all_ok) {
        printf("All results are correct !!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else {
        printf("Incorrect results\n");
    }

    
    //Part 5. Free the device and host memory

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    //Part 5 ends here
    

    return 0;
}
