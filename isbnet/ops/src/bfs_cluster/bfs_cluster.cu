/*
Ball Query with BatchIdx
Written by Li Jiang
All Rights Reserved 2020.
*/
#include "../cuda_utils.h"
#include "bfs_cluster.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ================================== ballquery_batch_p
 * ================================== */
__global__ void ballquery_batch_p_cuda_(int n, int meanActive, float radius,
                                        const float *xyz, const int *batch_idxs,
                                        const int *batch_offsets, int *idx,
                                        int *start_len, int *cumsum) {
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pt_idx >= n)
    return;

  start_len += (pt_idx * 2);
  int idx_temp[1000];

  float radius2 = radius * radius;
  float o_x = xyz[pt_idx * 3 + 0];
  float o_y = xyz[pt_idx * 3 + 1];
  float o_z = xyz[pt_idx * 3 + 2];

  int batch_idx = batch_idxs[pt_idx];
  int start = batch_offsets[batch_idx];
  int end = batch_offsets[batch_idx + 1];

  int cnt = 0;
  for (int k = start; k < end; k++) {
    float x = xyz[k * 3 + 0];
    float y = xyz[k * 3 + 1];
    float z = xyz[k * 3 + 2];
    float d2 =
        (o_x - x) * (o_x - x) + (o_y - y) * (o_y - y) + (o_z - z) * (o_z - z);
    if (d2 < radius2) {
      if (cnt < 1000) {
        idx_temp[cnt] = k;
      } else {
        break;
      }
      ++cnt;
    }
  }

  start_len[0] = atomicAdd(cumsum, cnt);
  start_len[1] = cnt;

  int thre = n * meanActive;
  if (start_len[0] >= thre)
    return;

  idx += start_len[0];
  if (start_len[0] + cnt >= thre)
    cnt = thre - start_len[0];

  for (int k = 0; k < cnt; k++) {
    idx[k] = idx_temp[k];
  }
}

__global__ void ballquery_batch_p_boxiou_cuda_(int n, int meanActive, float thresh_iou,
                                        const float *xyz_min, const float *xyz_max, const int *batch_idxs,
                                        const int *batch_offsets, int *idx,
                                        int *start_len, int *cumsum) {
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pt_idx >= n)
    return;

  start_len += (pt_idx * 2);
  int idx_temp[1000];



  float o_x1 = xyz_min[pt_idx * 3 + 0];
  float o_y1 = xyz_min[pt_idx * 3 + 1];
  float o_z1 = xyz_min[pt_idx * 3 + 2];

  float o_x2 = xyz_max[pt_idx * 3 + 0];
  float o_y2 = xyz_max[pt_idx * 3 + 1];
  float o_z2 = xyz_max[pt_idx * 3 + 2];

  // float o_x2 = xyz[pt_idx * 3 + 0];
  // float o_y2 = xyz[pt_idx * 3 + 1];
  // float o_z2 = xyz[pt_idx * 3 + 2];

  float pivot_area = (o_x2 - o_x1) * (o_y2 - o_y1) * (o_z2 - o_z1);

  int batch_idx = batch_idxs[pt_idx];
  int start = batch_offsets[batch_idx];
  int end = batch_offsets[batch_idx + 1];

  int cnt = 0;
  for (int k = start; k < end; k++) {
    float x1 = xyz_min[k * 3 + 0];
    float y1 = xyz_min[k * 3 + 1];
    float z1 = xyz_min[k * 3 + 2];

    float x2 = xyz_max[k * 3 + 0];
    float y2 = xyz_max[k * 3 + 1];
    float z2 = xyz_max[k * 3 + 2];

    // float x2 = xyz[k * 3 + 0];
    // float y2 = xyz[k * 3 + 1];
    // float z2 = xyz[k * 3 + 2];

    float cur_area = (x2 - x1) * (y2 - y1) * (z2 - z1);

    float x_upper = (x2 < o_x2) ? x2 : o_x2;
    float y_upper = (y2 < o_y2) ? y2 : o_y2;
    float z_upper = (z2 < o_z2) ? z2 : o_z2;

    float x_lower = (x1 > o_x1) ? x1 : o_x1;
    float y_lower = (y1 > o_y1) ? y1 : o_y1;
    float z_lower = (z1 > o_z1) ? z1 : o_z1;

    float range_x = (x_upper > x_lower) ? (x_upper - x_lower) : 0.0;
    float range_y = (y_upper > y_lower) ? (y_upper - y_lower) : 0.0;
    float range_z = (z_upper > z_lower) ? (z_upper - z_lower) : 0.0;

    float intersection_vol = range_x * range_y * range_z;
    float union_vol = cur_area + pivot_area - intersection_vol;

    float iou = intersection_vol / union_vol;

    // float d2 =
    //     (o_x2 - x2) * (o_x2 - x2) + (o_y2 - y2) * (o_y2 - y2) + (o_z2 - z2) * (o_z2 - z2);

    if (iou >= thresh_iou) {
      if (cnt < 1000) {
        idx_temp[cnt] = k;
      } else {
        break;
      }
      ++cnt;
    }
  }

  start_len[0] = atomicAdd(cumsum, cnt);
  start_len[1] = cnt;

  int thre = n * meanActive;
  if (start_len[0] >= thre)
    return;

  idx += start_len[0];
  if (start_len[0] + cnt >= thre)
    cnt = thre - start_len[0];

  for (int k = 0; k < cnt; k++) {
    idx[k] = idx_temp[k];
  }
}


int ballquery_batch_p_cuda(int n, int meanActive, float radius,
                           const float *xyz, const int *batch_idxs,
                           const int *batch_offsets, int *idx, int *start_len,
                           cudaStream_t stream) {
  // param xyz: (n, 3)
  // param batch_idxs: (n)
  // param batch_offsets: (B + 1)
  // output idx: (n * meanActive) dim 0 for number of points in the ball, idx in
  // n
  // output start_len: (n, 2), int

  cudaError_t err;

  dim3 blocks(DIVUP(n, MAX_THREADS_PER_BLOCK));
  dim3 threads(MAX_THREADS_PER_BLOCK);

  int cumsum = 0;
  int *p_cumsum;
  cudaMalloc((void **)&p_cumsum, sizeof(int));
  cudaMemcpy(p_cumsum, &cumsum, sizeof(int), cudaMemcpyHostToDevice);

  ballquery_batch_p_cuda_<<<blocks, threads, 0, stream>>>(
      n, meanActive, radius, xyz, batch_idxs, batch_offsets, idx, start_len,
      p_cumsum);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  cudaMemcpy(&cumsum, p_cumsum, sizeof(int), cudaMemcpyDeviceToHost);
  return cumsum;
}

int ballquery_batch_p_boxiou_cuda(int n, int meanActive, float radius,
                           const float *xyz_min, const float *xyz_max, const int *batch_idxs,
                           const int *batch_offsets, int *idx, int *start_len,
                           cudaStream_t stream) {
  // param xyz: (n, 3)
  // param batch_idxs: (n)
  // param batch_offsets: (B + 1)
  // output idx: (n * meanActive) dim 0 for number of points in the ball, idx in
  // n
  // output start_len: (n, 2), int

  cudaError_t err;

  dim3 blocks(DIVUP(n, MAX_THREADS_PER_BLOCK));
  dim3 threads(MAX_THREADS_PER_BLOCK);

  int cumsum = 0;
  int *p_cumsum;
  cudaMalloc((void **)&p_cumsum, sizeof(int));
  cudaMemcpy(p_cumsum, &cumsum, sizeof(int), cudaMemcpyHostToDevice);

  ballquery_batch_p_boxiou_cuda_<<<blocks, threads, 0, stream>>>(
      n, meanActive, radius, xyz_min, xyz_max, batch_idxs, batch_offsets, idx, start_len,
      p_cumsum);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  cudaMemcpy(&cumsum, p_cumsum, sizeof(int), cudaMemcpyDeviceToHost);
  return cumsum;
}




