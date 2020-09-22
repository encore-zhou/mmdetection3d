#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK_SQRT 8
#define THREADS_PER_BLOCK (THREADS_PER_BLOCK_SQRT * THREADS_PER_BLOCK_SQRT)
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void feat_distance_kernel(int b, int n, int m, int c,
                                     const float *__restrict__ feat_a,
                                     const float *__restrict__ feat_b,
                                     float *__restrict__ distance) {
  // feat_a: (B, N, C)
  // feat_b: (B, M, C)
  // output:
  //      distance: (B, N, M)
  int bs_idx = blockIdx.z;
  if (bs_idx >= b) return;
  int feat_a_idx = blockIdx.y * blockDim.y + threadIdx.y;
  int feat_b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (feat_a_idx >= n || feat_b_idx >= m) return;

  feat_a += bs_idx * n * c + feat_a_idx * c;
  feat_b += bs_idx * m * c + feat_b_idx * c;
  distance += bs_idx * n * m + feat_a_idx * m + feat_b_idx;
  float dist_tmp = 0;
  for (int j = 0; j < c; ++j) {
    dist_tmp += (feat_a[j] - feat_b[j]) * (feat_a[j] - feat_b[j]);
  }
  distance[0] = dist_tmp;
}

void feat_distance_kernel_launcher(int b, int n, int m, int c,
                                   const float *feat_a, const float *feat_b,
                                   float *distance, cudaStream_t stream) {
  // feat_a: (B, N, C)
  // feat_b: (B, M, C)
  // output:
  //      distance: (B, N, M)

  cudaError_t err;
  dim3 blocks(DIVUP(m, THREADS_PER_BLOCK_SQRT),
              DIVUP(n, THREADS_PER_BLOCK_SQRT), b);
  dim3 threads(THREADS_PER_BLOCK_SQRT, THREADS_PER_BLOCK_SQRT);

  feat_distance_kernel<<<blocks, threads, 0, stream>>>(b, n, m, c, feat_a,
                                                       feat_b, distance);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
