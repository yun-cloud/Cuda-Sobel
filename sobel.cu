#include <cstdlib>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iostream>
#include "utils/utils.h"

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8

#define TILE_WIDTH 16
#define MASK_WIDTH 5
// (MASK_WIDTH - 1) / 2
#define MASK_RADIUS 2
// TILE_WIDTH + MASK_WIDTH - 1
#define SHARE_MEMORY_WIDTH 20
#define CHANNEL_NUM 3

#define CHK(ans)                                                               \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "[GPUassert] %s(%d): %s\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__constant__ int filter[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                       { -2, -8, -12, -8, -2 },
                                       { 0, 0, 0, 0, 0 },
                                       { 2, 8, 12, 8, 2 },
                                       { 1, 4, 6, 4, 1 } },
                                     { { -1, -2, 0, 2, 1 },
                                       { -4, -8, 0, 8, 4 },
                                       { -6, -12, 0, 12, 6 },
                                       { -4, -8, 0, 8, 4 },
                                       { -1, -2, 0, 2, 1 } } };

inline __device__ int bound_check(int val, int lower, int upper) {
  if (val >= lower && val < upper)
    return 1;
  else
    return 0;
}

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height,
                      unsigned width, unsigned channels) {
  __shared__ unsigned char sm
      [CHANNEL_NUM * SHARE_MEMORY_WIDTH * SHARE_MEMORY_WIDTH];
  for (int c = 0; c < CHANNEL_NUM; ++c) {
    // ceil(SHARE_MEMORY_WIDTH^2 / TILE_WIDTH^2): how many times you need to load
    // First batch load
    int ID = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int dest_y = ID / SHARE_MEMORY_WIDTH;
    int dest_x = ID % SHARE_MEMORY_WIDTH;
    int src_y = blockIdx.y * TILE_WIDTH + dest_y - MASK_RADIUS;
    int src_x = blockIdx.x * TILE_WIDTH + dest_x - MASK_RADIUS;
    int dest_index = (dest_y * SHARE_MEMORY_WIDTH + dest_x) * CHANNEL_NUM + c;
    int src_index = (src_y * width + src_x) * CHANNEL_NUM + c;
    if (dest_y < SHARE_MEMORY_WIDTH) {
      if (bound_check(src_y, 0, height) && bound_check(src_x, 0, width)) {
        sm[dest_index] = s[src_index];
      } else {
        sm[dest_index] = 0;
      }
    }

    // second batch load
    ID = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    dest_y = ID / SHARE_MEMORY_WIDTH;
    dest_x = ID % SHARE_MEMORY_WIDTH;
    src_y = blockIdx.y * TILE_WIDTH + dest_y - MASK_RADIUS;
    src_x = blockIdx.x * TILE_WIDTH + dest_x - MASK_RADIUS;
    dest_index = (dest_y * SHARE_MEMORY_WIDTH + dest_x) * CHANNEL_NUM + c;
    src_index = (src_y * width + src_x) * CHANNEL_NUM + c;
    if (dest_y < SHARE_MEMORY_WIDTH) {
      if (bound_check(src_y, 0, height) && bound_check(src_x, 0, width)) {
        sm[dest_index] = s[src_index];
      } else {
        sm[dest_index] = 0;
      }
    }
  }
  __syncthreads();

  float val[Z][3];

  int src_y = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int src_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int y = threadIdx.y;
  int x = threadIdx.x;
  {
    if (bound_check(src_y, 0, height) && bound_check(src_x, 0, width)) {
      /* Z axis of filter */
      for (int i = 0; i < Z; ++i) {

        val[i][2] = 0.;
        val[i][1] = 0.;
        val[i][0] = 0.;

        /* Y and X axis of filter */
        for (int v = 0; v < MASK_WIDTH; ++v) {
          for (int u = 0; u < MASK_WIDTH; ++u) {
            {
              const unsigned char R =
                  sm[channels * (SHARE_MEMORY_WIDTH * (y + v) + (x + u)) + 2];
              const unsigned char G =
                  sm[channels * (SHARE_MEMORY_WIDTH * (y + v) + (x + u)) + 1];
              const unsigned char B =
                  sm[channels * (SHARE_MEMORY_WIDTH * (y + v) + (x + u)) + 0];
              val[i][2] += R * filter[i][u][v];
              val[i][1] += G * filter[i][u][v];
              val[i][0] += B * filter[i][u][v];
            }
          }
        }
      }
      float totalR = 0.;
      float totalG = 0.;
      float totalB = 0.;
      for (int i = 0; i < Z; ++i) {
        totalR += val[i][2] * val[i][2];
        totalG += val[i][1] * val[i][1];
        totalB += val[i][0] * val[i][0];
      }
      totalR = sqrt(totalR) / SCALE;
      totalG = sqrt(totalG) / SCALE;
      totalB = sqrt(totalB) / SCALE;
      const unsigned char cR = (totalR > 255.) ? 255 : totalR;
      const unsigned char cG = (totalG > 255.) ? 255 : totalG;
      const unsigned char cB = (totalB > 255.) ? 255 : totalB;
      t[channels * (width * src_y + src_x) + 2] = cR;
      t[channels * (width * src_y + src_x) + 1] = cG;
      t[channels * (width * src_y + src_x) + 0] = cB;
    }
  }
}

int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned height, width, channels;
  unsigned char *src = NULL, *dst;
  unsigned char *dsrc, *ddst;
  cudaEvent_t kernel_begin, kernel_end;
  CHK(cudaEventCreate(&kernel_begin));
  CHK(cudaEventCreate(&kernel_end));

  /* read the image to src, and get height, width, channels */
  if (read_png(argv[1], &src, &height, &width, &channels)) {
    std::cerr << "Error in read png" << std::endl;
    return -1;
  }

  dst = (unsigned char *)malloc(height * width * channels *
                                sizeof(unsigned char));
  CHK(cudaHostRegister(src, height * width * channels * sizeof(unsigned char),
                       cudaHostRegisterDefault));

  // cudaMalloc(...) for device src and device dst
  CHK(cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char)));
  CHK(cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char)));

  // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
  CHK(cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char),
                 cudaMemcpyHostToDevice));

  // decide to use how many blocks and threads
  dim3 num_threads(TILE_WIDTH, TILE_WIDTH);
  dim3 num_blocks(std::ceil((float)width / TILE_WIDTH),
                  std::ceil((float)height / TILE_WIDTH));

  // launch cuda kernel
  CHK(cudaEventRecord(kernel_begin));
  sobel << <num_blocks, num_threads>>> (dsrc, ddst, height, width, channels);
  CHK(cudaEventRecord(kernel_end));

  // cudaMemcpy(...) copy result image to host
  CHK(cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char),
                 cudaMemcpyDeviceToHost));

  CHK(cudaEventSynchronize(kernel_end));
  float kernel_time = 0;
  CHK(cudaEventElapsedTime(&kernel_time, kernel_begin, kernel_end));
  std::cout << "Kernel execution time: " << kernel_time << "ms\n";

  write_png(argv[2], dst, height, width, channels);
  free(src);
  free(dst);
  CHK(cudaFree(dsrc));
  CHK(cudaFree(ddst));
  return 0;
}
