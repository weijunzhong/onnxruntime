// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is generated by compile_sparse_attention.py using triton AoT compiler

#pragma once
#include "contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace sparse_attention_v1 {

// launcher for: sparse_attention_bf16_sm90_16x0x64x0x64x2_warps1xstages3
Status sparse_attention_bf16_sm90_0e80ced5(SparseAttentionParams& params);

Status sparse_attention_bf16_sm90_16x0x64x0x64x2_warps1xstages3(SparseAttentionParams& params) {
  return sparse_attention_bf16_sm90_0e80ced5(params);
}

// load for: sparse_attention_bf16_sm90_16x0x64x0x64x2_warps1xstages3
void load_sparse_attention_bf16_sm90_0e80ced5();
void load_sparse_attention_bf16_sm90_16x0x64x0x64x2_warps1xstages3() {
  load_sparse_attention_bf16_sm90_0e80ced5();
}

// unload for: sparse_attention_bf16_sm90_16x0x64x0x64x2_warps1xstages3
void unload_sparse_attention_bf16_sm90_0e80ced5();
void unload_sparse_attention_bf16_sm90_16x0x64x0x64x2_warps1xstages3() {
  unload_sparse_attention_bf16_sm90_0e80ced5();
}

// launcher for: sparse_attention_bf16_sm90_16x0x64x1x64x2_warps1xstages3
Status sparse_attention_bf16_sm90_e13c9217(SparseAttentionParams& params);

Status sparse_attention_bf16_sm90_16x0x64x1x64x2_warps1xstages3(SparseAttentionParams& params) {
  return sparse_attention_bf16_sm90_e13c9217(params);
}

// load for: sparse_attention_bf16_sm90_16x0x64x1x64x2_warps1xstages3
void load_sparse_attention_bf16_sm90_e13c9217();
void load_sparse_attention_bf16_sm90_16x0x64x1x64x2_warps1xstages3() {
  load_sparse_attention_bf16_sm90_e13c9217();
}

// unload for: sparse_attention_bf16_sm90_16x0x64x1x64x2_warps1xstages3
void unload_sparse_attention_bf16_sm90_e13c9217();
void unload_sparse_attention_bf16_sm90_16x0x64x1x64x2_warps1xstages3() {
  unload_sparse_attention_bf16_sm90_e13c9217();
}

// launcher for: sparse_attention_bf16_sm90_16x1x64x0x64x2_warps1xstages3
Status sparse_attention_bf16_sm90_37f6ea0d(SparseAttentionParams& params);

Status sparse_attention_bf16_sm90_16x1x64x0x64x2_warps1xstages3(SparseAttentionParams& params) {
  return sparse_attention_bf16_sm90_37f6ea0d(params);
}

// load for: sparse_attention_bf16_sm90_16x1x64x0x64x2_warps1xstages3
void load_sparse_attention_bf16_sm90_37f6ea0d();
void load_sparse_attention_bf16_sm90_16x1x64x0x64x2_warps1xstages3() {
  load_sparse_attention_bf16_sm90_37f6ea0d();
}

// unload for: sparse_attention_bf16_sm90_16x1x64x0x64x2_warps1xstages3
void unload_sparse_attention_bf16_sm90_37f6ea0d();
void unload_sparse_attention_bf16_sm90_16x1x64x0x64x2_warps1xstages3() {
  unload_sparse_attention_bf16_sm90_37f6ea0d();
}

// launcher for: sparse_attention_bf16_sm90_16x1x64x1x64x2_warps1xstages3
Status sparse_attention_bf16_sm90_f7dddbf9(SparseAttentionParams& params);

Status sparse_attention_bf16_sm90_16x1x64x1x64x2_warps1xstages3(SparseAttentionParams& params) {
  return sparse_attention_bf16_sm90_f7dddbf9(params);
}

// load for: sparse_attention_bf16_sm90_16x1x64x1x64x2_warps1xstages3
void load_sparse_attention_bf16_sm90_f7dddbf9();
void load_sparse_attention_bf16_sm90_16x1x64x1x64x2_warps1xstages3() {
  load_sparse_attention_bf16_sm90_f7dddbf9();
}

// unload for: sparse_attention_bf16_sm90_16x1x64x1x64x2_warps1xstages3
void unload_sparse_attention_bf16_sm90_f7dddbf9();
void unload_sparse_attention_bf16_sm90_16x1x64x1x64x2_warps1xstages3() {
  unload_sparse_attention_bf16_sm90_f7dddbf9();
}

// launcher for: sparse_attention_bf16_sm90_64x0x64x0x64x2_warps4xstages3
Status sparse_attention_bf16_sm90_eb17c351(SparseAttentionParams& params);

Status sparse_attention_bf16_sm90_64x0x64x0x64x2_warps4xstages3(SparseAttentionParams& params) {
  return sparse_attention_bf16_sm90_eb17c351(params);
}

// load for: sparse_attention_bf16_sm90_64x0x64x0x64x2_warps4xstages3
void load_sparse_attention_bf16_sm90_eb17c351();
void load_sparse_attention_bf16_sm90_64x0x64x0x64x2_warps4xstages3() {
  load_sparse_attention_bf16_sm90_eb17c351();
}

// unload for: sparse_attention_bf16_sm90_64x0x64x0x64x2_warps4xstages3
void unload_sparse_attention_bf16_sm90_eb17c351();
void unload_sparse_attention_bf16_sm90_64x0x64x0x64x2_warps4xstages3() {
  unload_sparse_attention_bf16_sm90_eb17c351();
}

// launcher for: sparse_attention_bf16_sm90_64x0x64x1x64x2_warps4xstages3
Status sparse_attention_bf16_sm90_fc0bc8a9(SparseAttentionParams& params);

Status sparse_attention_bf16_sm90_64x0x64x1x64x2_warps4xstages3(SparseAttentionParams& params) {
  return sparse_attention_bf16_sm90_fc0bc8a9(params);
}

// load for: sparse_attention_bf16_sm90_64x0x64x1x64x2_warps4xstages3
void load_sparse_attention_bf16_sm90_fc0bc8a9();
void load_sparse_attention_bf16_sm90_64x0x64x1x64x2_warps4xstages3() {
  load_sparse_attention_bf16_sm90_fc0bc8a9();
}

// unload for: sparse_attention_bf16_sm90_64x0x64x1x64x2_warps4xstages3
void unload_sparse_attention_bf16_sm90_fc0bc8a9();
void unload_sparse_attention_bf16_sm90_64x0x64x1x64x2_warps4xstages3() {
  unload_sparse_attention_bf16_sm90_fc0bc8a9();
}

// launcher for: sparse_attention_bf16_sm90_64x1x64x0x64x2_warps4xstages3
Status sparse_attention_bf16_sm90_2f5e24aa(SparseAttentionParams& params);

Status sparse_attention_bf16_sm90_64x1x64x0x64x2_warps4xstages3(SparseAttentionParams& params) {
  return sparse_attention_bf16_sm90_2f5e24aa(params);
}

// load for: sparse_attention_bf16_sm90_64x1x64x0x64x2_warps4xstages3
void load_sparse_attention_bf16_sm90_2f5e24aa();
void load_sparse_attention_bf16_sm90_64x1x64x0x64x2_warps4xstages3() {
  load_sparse_attention_bf16_sm90_2f5e24aa();
}

// unload for: sparse_attention_bf16_sm90_64x1x64x0x64x2_warps4xstages3
void unload_sparse_attention_bf16_sm90_2f5e24aa();
void unload_sparse_attention_bf16_sm90_64x1x64x0x64x2_warps4xstages3() {
  unload_sparse_attention_bf16_sm90_2f5e24aa();
}

// launcher for: sparse_attention_bf16_sm90_64x1x64x1x64x2_warps4xstages3
Status sparse_attention_bf16_sm90_d7dba852(SparseAttentionParams& params);

Status sparse_attention_bf16_sm90_64x1x64x1x64x2_warps4xstages3(SparseAttentionParams& params) {
  return sparse_attention_bf16_sm90_d7dba852(params);
}

// load for: sparse_attention_bf16_sm90_64x1x64x1x64x2_warps4xstages3
void load_sparse_attention_bf16_sm90_d7dba852();
void load_sparse_attention_bf16_sm90_64x1x64x1x64x2_warps4xstages3() {
  load_sparse_attention_bf16_sm90_d7dba852();
}

// unload for: sparse_attention_bf16_sm90_64x1x64x1x64x2_warps4xstages3
void unload_sparse_attention_bf16_sm90_d7dba852();
void unload_sparse_attention_bf16_sm90_64x1x64x1x64x2_warps4xstages3() {
  unload_sparse_attention_bf16_sm90_d7dba852();
}

typedef Status (*kernel_func_t)(SparseAttentionParams& params);
kernel_func_t sparse_attention_bf16_sm90_kernels[] = {
    sparse_attention_bf16_sm90_16x0x64x0x64x2_warps1xstages3,
    sparse_attention_bf16_sm90_16x0x64x1x64x2_warps1xstages3,
    sparse_attention_bf16_sm90_16x1x64x0x64x2_warps1xstages3,
    sparse_attention_bf16_sm90_16x1x64x1x64x2_warps1xstages3,
    sparse_attention_bf16_sm90_64x0x64x0x64x2_warps4xstages3,
    sparse_attention_bf16_sm90_64x0x64x1x64x2_warps4xstages3,
    sparse_attention_bf16_sm90_64x1x64x0x64x2_warps4xstages3,
    sparse_attention_bf16_sm90_64x1x64x1x64x2_warps4xstages3,
};

int sparse_attention_bf16_sm90_get_num_algos(void) {
  return (int)sizeof(sparse_attention_bf16_sm90_kernels);
}

Status sparse_attention_bf16_sm90(SparseAttentionParams& params, int algo_id) {
  assert(algo_id < (int)sizeof(sparse_attention_bf16_sm90_kernels));
  return sparse_attention_bf16_sm90_kernels[algo_id](params);
}

void load_sparse_attention_bf16_sm90(void) {
  load_sparse_attention_bf16_sm90_16x0x64x0x64x2_warps1xstages3();
  load_sparse_attention_bf16_sm90_16x0x64x1x64x2_warps1xstages3();
  load_sparse_attention_bf16_sm90_16x1x64x0x64x2_warps1xstages3();
  load_sparse_attention_bf16_sm90_16x1x64x1x64x2_warps1xstages3();
  load_sparse_attention_bf16_sm90_64x0x64x0x64x2_warps4xstages3();
  load_sparse_attention_bf16_sm90_64x0x64x1x64x2_warps4xstages3();
  load_sparse_attention_bf16_sm90_64x1x64x0x64x2_warps4xstages3();
  load_sparse_attention_bf16_sm90_64x1x64x1x64x2_warps4xstages3();
}

void unload_sparse_attention_bf16_sm90(void) {
  unload_sparse_attention_bf16_sm90_16x0x64x0x64x2_warps1xstages3();
  unload_sparse_attention_bf16_sm90_16x0x64x1x64x2_warps1xstages3();
  unload_sparse_attention_bf16_sm90_16x1x64x0x64x2_warps1xstages3();
  unload_sparse_attention_bf16_sm90_16x1x64x1x64x2_warps1xstages3();
  unload_sparse_attention_bf16_sm90_64x0x64x0x64x2_warps4xstages3();
  unload_sparse_attention_bf16_sm90_64x0x64x1x64x2_warps4xstages3();
  unload_sparse_attention_bf16_sm90_64x1x64x0x64x2_warps4xstages3();
  unload_sparse_attention_bf16_sm90_64x1x64x1x64x2_warps4xstages3();
}

Status sparse_attention_bf16_sm90_default(SparseAttentionParams& params) {
  return sparse_attention_bf16_sm90(params, 0);
}

}  // namespace sparse_attention_v1
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
