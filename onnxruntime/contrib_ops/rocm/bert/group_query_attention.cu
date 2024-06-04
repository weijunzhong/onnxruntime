// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "ck_tile/core/numeric/integer.hpp"
#include "core/providers/rocm/rocm_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/rocm/bert/group_query_attention.h"
#include "contrib_ops/rocm/bert/group_query_attention_helper.h"
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"

#ifdef USE_COMPOSABLE_KERNEL_CK_TILE
#include "fmha_fwd.hpp"
#endif

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                       \
      GroupQueryAttention,                                             \
      kMSDomain,                                                       \
      1,                                                               \
      T,                                                               \
      kRocmExecutionProvider,                                          \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()) \
          .MayInplace(3, 1)                                            \
          .MayInplace(4, 2)                                            \
          .InputMemoryType(OrtMemTypeCPUInput, 6),                     \
      GroupQueryAttention<T>);

// REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
// REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
std::string GetCkFmhaDataTypeString();

template <>
std::string GetCkFmhaDataTypeString<MLFloat16>() {
  return "fp16";
}

template <>
std::string GetCkFmhaDataTypeString<BFloat16>() {
  return "bf16";
}

template <typename T>
GroupQueryAttention<T>::GroupQueryAttention(const OpKernelInfo& info)
    : RocmKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  is_past_bsnh_ = false;
  is_unidirectional_ = true;
  local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
Status GroupQueryAttention<T>::ComputeInternal(OpKernelContext* ctx) const {
  auto hip_stream = static_cast<hipStream_t>(ctx->GetComputeStream()->GetHandle());
  const Tensor* query = ctx->Input<Tensor>(0);
  const Tensor* key = ctx->Input<Tensor>(1);
  const Tensor* value = ctx->Input<Tensor>(2);
  const Tensor* past_key = ctx->Input<Tensor>(3);
  const Tensor* past_value = ctx->Input<Tensor>(4);
  const Tensor* seqlens_k = ctx->Input<Tensor>(5);
  const Tensor* total_seqlen = ctx->Input<Tensor>(6);
  const Tensor* cos_cache = ctx->Input<Tensor>(7);
  const Tensor* sin_cache = ctx->Input<Tensor>(8);

  auto& device_prop = GetDeviceProp();
  GroupQueryAttentionParameters parameters;
  using HipT = typename ToHipType<T>::MappedType;

  const int max_thr_per_blk = device_prop.maxThreadsPerBlock;

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                cos_cache,
                                                                sin_cache,
                                                                &parameters,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                seqlens_k,
                                                                total_seqlen,
                                                                is_past_bsnh_,
                                                                scale_,
                                                                max_thr_per_blk));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  AttentionQkvFormat past_kv_format = parameters.past_kv_format;

  parameters.local_window_size = local_window_size_;
  parameters.is_unidirectional = is_unidirectional_;
  // parameters.zeros_count = kZerosCount;
  // parameters.zero_ptr = zeros_.get();
  // parameters.left_padding = left_padding_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache must be passed to GroupQueryAttention when do_rotary = 1");
  }

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size);
  Tensor* output = ctx->Output(0, output_shape);
  Strides output_strides = Strides::BSNHMemory(batch_size, sequence_length, num_heads, head_size);

  int4 past_shape;
  std::vector<int64_t> present_dims;
  Strides present_strides;
  Strides past_strides;
  if (past_kv_format == AttentionQkvFormat::Q_K_V_BSNH) {
    past_shape = {
        batch_size, parameters.seqlen_past_kv_cache, kv_num_heads, head_size};
    past_strides = Strides::BSNHMemory(
        batch_size, parameters.seqlen_past_kv_cache, kv_num_heads, head_size);
    present_dims = {
        batch_size, parameters.seqlen_present_kv_cache, kv_num_heads, head_size};
    present_strides = Strides::BSNHMemory(
        batch_size, parameters.seqlen_present_kv_cache, kv_num_heads, head_size);
  } else {  // BNSH
    past_shape = {
        batch_size, kv_num_heads, parameters.seqlen_past_kv_cache, head_size};
    past_strides = Strides::BSNHMemory(
        batch_size, kv_num_heads, parameters.seqlen_past_kv_cache, head_size);
    present_dims = {
        batch_size, kv_num_heads, parameters.seqlen_present_kv_cache, head_size};
    present_strides = Strides::BNSHMemory(
        batch_size, kv_num_heads, parameters.seqlen_present_kv_cache, head_size);
  }
  TensorShape present_shape(present_dims);
  Tensor* present_key = ctx->Output(1, present_shape);
  Tensor* present_value = ctx->Output(2, present_shape);

  Strides query_strides;
  const void* query_ptr = query->DataRaw();
  const HipT* key_ptr;
  const HipT* value_ptr;
  if (!parameters.is_packed_qkv) {
    query_strides = Strides::BSNHMemory(batch_size, sequence_length, num_heads, head_size);
    key_ptr = reinterpret_cast<const HipT*>(key->DataRaw());
    value_ptr = reinterpret_cast<const HipT*>(value->DataRaw());
  } else {
    query_strides = Strides::BSNHMemory(batch_size, sequence_length, num_heads + 2 * kv_num_heads, head_size);
    const size_t key_offset = static_cast<size_t>(num_heads * head_size);
    const size_t value_offset = static_cast<size_t>(kv_num_heads * head_size);
    key_ptr = reinterpret_cast<const HipT*>(query_ptr) + key_offset;
    value_ptr = reinterpret_cast<const HipT*>(key_ptr) + value_offset;
  }

  // build present kv cache
  auto* present_key_ptr = reinterpret_cast<HipT*>(present_key->MutableDataRaw());
  auto* present_value_ptr = reinterpret_cast<HipT*>(present_value->MutableDataRaw());
  int4 kv_shape{batch_size, kv_sequence_length, kv_num_heads, head_size};
  auto kv_strides = Strides::BSNHMemory(batch_size, kv_sequence_length, kv_num_heads, head_size);
  if (parameters.is_prompt) {
    // copy prompt kv to present kv
    ORT_RETURN_IF_ERROR(LaunchStridedCopy(hip_stream, key_ptr, kv_shape, kv_strides.ForBNSHCoord(),
                                          present_key_ptr, present_strides.ForBNSHCoord(), max_thr_per_blk));
    ORT_RETURN_IF_ERROR(LaunchStridedCopy(hip_stream, value_ptr, kv_shape, kv_strides.ForBNSHCoord(),
                                          present_value_ptr, present_strides.ForBNSHCoord(), max_thr_per_blk));
  } else {
    if (!parameters.kv_share_buffer) {
      // copy past to present
      const auto* past_key_ptr = reinterpret_cast<const HipT*>(past_key->DataRaw());
      const auto* past_value_ptr = reinterpret_cast<const HipT*>(past_value->DataRaw());
      ORT_RETURN_IF_ERROR(LaunchStridedCopy(hip_stream, past_key_ptr, past_shape, past_strides.ForBNSHCoord(),
                                            present_key_ptr, present_strides.ForBNSHCoord(), max_thr_per_blk));
      ORT_RETURN_IF_ERROR(LaunchStridedCopy(hip_stream, past_value_ptr, past_shape, past_strides.ForBNSHCoord(),
                                            present_value_ptr, present_strides.ForBNSHCoord(), max_thr_per_blk));
    }
    // then append new kv to present
    // FIXME: present_key_ptr and present_value_ptr offset
    ORT_RETURN_IF_ERROR(LaunchStridedCopy(hip_stream, key_ptr, kv_shape, kv_strides.ForBNSHCoord(),
                                          present_key_ptr + present_strides.OffsetAt(0, 0, kv_sequence_length, 0), present_strides.ForBNSHCoord(), max_thr_per_blk));
    ORT_RETURN_IF_ERROR(LaunchStridedCopy(hip_stream, value_ptr, kv_shape, kv_strides.ForBNSHCoord(),
                                          present_value_ptr + present_strides.OffsetAt(0, 0, kv_sequence_length, 0), present_strides.ForBNSHCoord(), max_thr_per_blk));
  }
  static_assert(std::is_same_v<ck_tile::index_t, int32_t>);

  ck_tile::index_t shape_seqlen_q = batch_size;
  ck_tile::index_t shape_seqlen_k = batch_size;

  // TODO:
  mask_enum mask_type = mask_enum::no_mask;
  bias_enum bias_type = bias_enum::no_bias;
  mask_info mask = mask_info::decode("0", sequence_length, kv_sequence_length);

  // std::cout << "q:" << query->DataRaw() << ", k:" << present_key->DataRaw()

  fmha_fwd_args args{
      query->DataRaw(),
      present_key->DataRaw(),
      present_value->DataRaw(),
      nullptr,  // bias, alibi/element
      nullptr,  // lse, logsumexp buffer
      output->MutableDataRaw(),
      // if seqlen_k_ptr != nullptr, current seqlen is seqlen_k_ptr[cur_batch],
      // otherwise is seqstart_k_ptr[cur_batch+1] - seqstart_k_ptr[cur_batch]
      nullptr,                                     // seqstart_q_ptr
      nullptr,                                     // seqstart_k_ptr
      seqlens_k ? seqlens_k->DataRaw() : nullptr,  // seqlen_k_ptr
      shape_seqlen_q,                              // seqlen_q
      shape_seqlen_k,                              // seqlen_k
      parameters.batch_size,                       // batch
      parameters.sequence_length,                  // max_seqlen_q
      parameters.head_size,                        // hdim_q
      parameters.head_size,                        // hdim_v
      parameters.num_heads,
      parameters.kv_num_heads,
      parameters.scale,
      1.0f,                                                                     // scale_p of squant, useless
      1.0f,                                                                     // scale_o of squant, useless
      static_cast<ck_tile::index_t>(query_strides.strides_for_bnsh_coord.z),    // stride_q, to be regarded as stride of dim S
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.z),  // stride_k, to be regarded as stride of dim S
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.z),  // stride_v, to be regarded as stride of dim S
      shape_seqlen_k,                                                           // stride_bias, if alibi, b*h need set this to h, 1*h need set this to 0
      static_cast<ck_tile::index_t>(output_strides.strides_for_bnsh_coord.z),   // stride_o, to be regarded as stride of dim S
      static_cast<ck_tile::index_t>(query_strides.strides_for_bnsh_coord.y),    // nhead_stride_q, to be regarded as stride of dim N
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.y),  // nhead_stride_k, to be regarded as stride of dim N
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.y),  // nhead_stride_v, to be regarded as stride of dim N
      0,                                                                        // nhead_stride_bias
      shape_seqlen_q,                                                           // nhead_stride_lse
      static_cast<ck_tile::index_t>(output_strides.strides_for_bnsh_coord.y),   // batch_stride_o, to be regarded as stride of dim B
      static_cast<ck_tile::index_t>(query_strides.strides_for_bnsh_coord.x),    // batch_stride_q, to be regarded as stride of dim B
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.x),  // batch_stride_k, to be regarded as stride of dim B
      static_cast<ck_tile::index_t>(present_strides.strides_for_bnsh_coord.x),  // batch_stride_v, to be regarded as stride of dim B
      0,                                                                        // batch_stride_bias
      num_heads * shape_seqlen_q,                                               // batch_stride_lse
      static_cast<ck_tile::index_t>(output_strides.strides_for_bnsh_coord.x),   // batch_stride_o, to be regarded as stride of dim B
      mask.left,                                                                // window_size_left
      mask.right,                                                               // window_size_right
      static_cast<ck_tile::index_t>(mask.type)};

  fmha_fwd_traits traits{
      parameters.head_size,
      parameters.head_size,  // v head size
      GetCkFmhaDataTypeString<T>(),
      false,  // is_group_mode
      true,   // is_v_rowmajor ? dim is fastest : seq is fastest
      mask_type,
      bias_type,
      false,  // has_lse
      false,  // do_fp8_static_quant, aka, squant
  };

  ck_tile::stream_config stream_config{
      hip_stream,
      false  // time_kernel
  };

  auto duration = fmha_fwd(traits, args, stream_config);
  if (duration < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "fmha_fwd internal error");
  }
  HIP_RETURN_IF_ERROR(hipGetLastError());

  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
