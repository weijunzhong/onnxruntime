// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is generated by compile_sparse_attention.py using triton AoT compiler

#pragma once
#include "contrib_ops/cuda/sparse/sparse_attention_trition/sparse_attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// launcher for: sparse_attention_fp16_sm80_16x0x64x0x64x2_warps1xstages2
Status sparse_attention_fp16_sm80_a94506bb(SparseAttentionParams& params);

Status sparse_attention_fp16_sm80_16x0x64x0x64x2_warps1xstages2(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80_a94506bb(params);
}

// load for: sparse_attention_fp16_sm80_16x0x64x0x64x2_warps1xstages2
void load_sparse_attention_fp16_sm80_a94506bb();
void load_sparse_attention_fp16_sm80_16x0x64x0x64x2_warps1xstages2() {
  load_sparse_attention_fp16_sm80_a94506bb();
}

// unload for: sparse_attention_fp16_sm80_16x0x64x0x64x2_warps1xstages2
void unload_sparse_attention_fp16_sm80_a94506bb();
void unload_sparse_attention_fp16_sm80_16x0x64x0x64x2_warps1xstages2() {
  unload_sparse_attention_fp16_sm80_a94506bb();
}

// launcher for: sparse_attention_fp16_sm80_16x0x64x1x64x2_warps1xstages2
Status sparse_attention_fp16_sm80_f1e7ed8e(SparseAttentionParams& params);

Status sparse_attention_fp16_sm80_16x0x64x1x64x2_warps1xstages2(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80_f1e7ed8e(params);
}

// load for: sparse_attention_fp16_sm80_16x0x64x1x64x2_warps1xstages2
void load_sparse_attention_fp16_sm80_f1e7ed8e();
void load_sparse_attention_fp16_sm80_16x0x64x1x64x2_warps1xstages2() {
  load_sparse_attention_fp16_sm80_f1e7ed8e();
}

// unload for: sparse_attention_fp16_sm80_16x0x64x1x64x2_warps1xstages2
void unload_sparse_attention_fp16_sm80_f1e7ed8e();
void unload_sparse_attention_fp16_sm80_16x0x64x1x64x2_warps1xstages2() {
  unload_sparse_attention_fp16_sm80_f1e7ed8e();
}

// launcher for: sparse_attention_fp16_sm80_16x1x64x0x64x2_warps1xstages2
Status sparse_attention_fp16_sm80_5d10b453(SparseAttentionParams& params);

Status sparse_attention_fp16_sm80_16x1x64x0x64x2_warps1xstages2(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80_5d10b453(params);
}

// load for: sparse_attention_fp16_sm80_16x1x64x0x64x2_warps1xstages2
void load_sparse_attention_fp16_sm80_5d10b453();
void load_sparse_attention_fp16_sm80_16x1x64x0x64x2_warps1xstages2() {
  load_sparse_attention_fp16_sm80_5d10b453();
}

// unload for: sparse_attention_fp16_sm80_16x1x64x0x64x2_warps1xstages2
void unload_sparse_attention_fp16_sm80_5d10b453();
void unload_sparse_attention_fp16_sm80_16x1x64x0x64x2_warps1xstages2() {
  unload_sparse_attention_fp16_sm80_5d10b453();
}

// launcher for: sparse_attention_fp16_sm80_16x1x64x1x64x2_warps1xstages2
Status sparse_attention_fp16_sm80_b286eb9c(SparseAttentionParams& params);

Status sparse_attention_fp16_sm80_16x1x64x1x64x2_warps1xstages2(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80_b286eb9c(params);
}

// load for: sparse_attention_fp16_sm80_16x1x64x1x64x2_warps1xstages2
void load_sparse_attention_fp16_sm80_b286eb9c();
void load_sparse_attention_fp16_sm80_16x1x64x1x64x2_warps1xstages2() {
  load_sparse_attention_fp16_sm80_b286eb9c();
}

// unload for: sparse_attention_fp16_sm80_16x1x64x1x64x2_warps1xstages2
void unload_sparse_attention_fp16_sm80_b286eb9c();
void unload_sparse_attention_fp16_sm80_16x1x64x1x64x2_warps1xstages2() {
  unload_sparse_attention_fp16_sm80_b286eb9c();
}

// launcher for: sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2
Status sparse_attention_fp16_sm80_739da152(SparseAttentionParams& params);

Status sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80_739da152(params);
}

// load for: sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2
void load_sparse_attention_fp16_sm80_739da152();
void load_sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2() {
  load_sparse_attention_fp16_sm80_739da152();
}

// unload for: sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2
void unload_sparse_attention_fp16_sm80_739da152();
void unload_sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2() {
  unload_sparse_attention_fp16_sm80_739da152();
}

// launcher for: sparse_attention_fp16_sm80_64x0x64x1x64x2_warps4xstages2
Status sparse_attention_fp16_sm80_eb8740ba(SparseAttentionParams& params);

Status sparse_attention_fp16_sm80_64x0x64x1x64x2_warps4xstages2(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80_eb8740ba(params);
}

// load for: sparse_attention_fp16_sm80_64x0x64x1x64x2_warps4xstages2
void load_sparse_attention_fp16_sm80_eb8740ba();
void load_sparse_attention_fp16_sm80_64x0x64x1x64x2_warps4xstages2() {
  load_sparse_attention_fp16_sm80_eb8740ba();
}

// unload for: sparse_attention_fp16_sm80_64x0x64x1x64x2_warps4xstages2
void unload_sparse_attention_fp16_sm80_eb8740ba();
void unload_sparse_attention_fp16_sm80_64x0x64x1x64x2_warps4xstages2() {
  unload_sparse_attention_fp16_sm80_eb8740ba();
}

// launcher for: sparse_attention_fp16_sm80_64x1x64x0x64x2_warps4xstages2
Status sparse_attention_fp16_sm80_a105022b(SparseAttentionParams& params);

Status sparse_attention_fp16_sm80_64x1x64x0x64x2_warps4xstages2(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80_a105022b(params);
}

// load for: sparse_attention_fp16_sm80_64x1x64x0x64x2_warps4xstages2
void load_sparse_attention_fp16_sm80_a105022b();
void load_sparse_attention_fp16_sm80_64x1x64x0x64x2_warps4xstages2() {
  load_sparse_attention_fp16_sm80_a105022b();
}

// unload for: sparse_attention_fp16_sm80_64x1x64x0x64x2_warps4xstages2
void unload_sparse_attention_fp16_sm80_a105022b();
void unload_sparse_attention_fp16_sm80_64x1x64x0x64x2_warps4xstages2() {
  unload_sparse_attention_fp16_sm80_a105022b();
}

// launcher for: sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2
Status sparse_attention_fp16_sm80_f4a89c6a(SparseAttentionParams& params);

Status sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80_f4a89c6a(params);
}

// load for: sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2
void load_sparse_attention_fp16_sm80_f4a89c6a();
void load_sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2() {
  load_sparse_attention_fp16_sm80_f4a89c6a();
}

// unload for: sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2
void unload_sparse_attention_fp16_sm80_f4a89c6a();
void unload_sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2() {
  unload_sparse_attention_fp16_sm80_f4a89c6a();
}

typedef Status (*kernel_func_t)(SparseAttentionParams& params);
kernel_func_t sparse_attention_fp16_sm80_kernels[] = {
    sparse_attention_fp16_sm80_16x0x64x0x64x2_warps1xstages2,
    sparse_attention_fp16_sm80_16x0x64x1x64x2_warps1xstages2,
    sparse_attention_fp16_sm80_16x1x64x0x64x2_warps1xstages2,
    sparse_attention_fp16_sm80_16x1x64x1x64x2_warps1xstages2,
    sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2,
    sparse_attention_fp16_sm80_64x0x64x1x64x2_warps4xstages2,
    sparse_attention_fp16_sm80_64x1x64x0x64x2_warps4xstages2,
    sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2,
};

int sparse_attention_fp16_sm80_get_num_algos(void) {
  return (int)sizeof(sparse_attention_fp16_sm80_kernels);
}

Status sparse_attention_fp16_sm80(SparseAttentionParams& params, int algo_id) {
  assert(algo_id < (int)sizeof(sparse_attention_fp16_sm80_kernels));
  return sparse_attention_fp16_sm80_kernels[algo_id](params);
}

void load_sparse_attention_fp16_sm80(void) {
  load_sparse_attention_fp16_sm80_16x0x64x0x64x2_warps1xstages2();
  load_sparse_attention_fp16_sm80_16x0x64x1x64x2_warps1xstages2();
  load_sparse_attention_fp16_sm80_16x1x64x0x64x2_warps1xstages2();
  load_sparse_attention_fp16_sm80_16x1x64x1x64x2_warps1xstages2();
  load_sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2();
  load_sparse_attention_fp16_sm80_64x0x64x1x64x2_warps4xstages2();
  load_sparse_attention_fp16_sm80_64x1x64x0x64x2_warps4xstages2();
  load_sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2();
}

void unload_sparse_attention_fp16_sm80(void) {
  unload_sparse_attention_fp16_sm80_16x0x64x0x64x2_warps1xstages2();
  unload_sparse_attention_fp16_sm80_16x0x64x1x64x2_warps1xstages2();
  unload_sparse_attention_fp16_sm80_16x1x64x0x64x2_warps1xstages2();
  unload_sparse_attention_fp16_sm80_16x1x64x1x64x2_warps1xstages2();
  unload_sparse_attention_fp16_sm80_64x0x64x0x64x2_warps4xstages2();
  unload_sparse_attention_fp16_sm80_64x0x64x1x64x2_warps4xstages2();
  unload_sparse_attention_fp16_sm80_64x1x64x0x64x2_warps4xstages2();
  unload_sparse_attention_fp16_sm80_64x1x64x1x64x2_warps4xstages2();
}

Status sparse_attention_fp16_sm80_default(SparseAttentionParams& params) {
  return sparse_attention_fp16_sm80(params, 0);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
