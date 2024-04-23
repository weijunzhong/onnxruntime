# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Use triton AoT compiler to convert sparse_attention_triton.py to C source files including cubin and dispatcher.
# Example to use this script (Tested with triton 2.3.0 in Ubuntu 20.04):
#    python3 -m pip install triton==2.3.0
#    python3 compile_sparse_attention.py | sh
#
# Note that sparse_attention_kernel_*.* and sparse_attention_api.cc under this directory is modified from the generated files.

import math
from itertools import product


def generate_triton_compile_shell_script(dtype="fp16"):
    assert dtype in ["fp16", "bf16"]
    print("export TRITON_ROOT=$(pip show triton | grep Location | cut -d' ' -f2)")
    print('export ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)"')
    print("export SM=$(echo $ARCH | sed -e 's/\\.//g')")

    # Modify the compile.py to use custom template file template_h.txt and template_c.txt in current directory.
    # Also pass "block_m", "namespace_start" and "namespace_end" to the template
    print(
        'python -c "import sys;lines=sys.stdin.read();lines=lines.replace(\'template_path = Path(__file__).parent / f\\"compile.{ext}\\"\',\'template_path = f\\"compile_template_{ext}.txt\\"\');lines=lines.replace(\'\\"_placeholder\\": \\"\\",\', \'\\"_placeholder\\": \\"\\",\\n        \\"block_m\\": list(constants.values())[0],\');print(lines)" < ${TRITON_ROOT}/triton/tools/compile.py > compile.py'
    )

    out_dir = f"trition_cubin_{dtype}"
    print(f"rm -rf {out_dir}")
    print(f"mkdir -p {out_dir}")

    block_n_values = [64]
    block_d_values = [64]
    num_block_d_values = [2]
    even_m_values = [True, False]
    even_n_values = [True, False]

    for block_n, block_d, num_blocks_d, even_m, even_n in product(
        block_n_values, block_d_values, num_block_d_values, even_m_values, even_n_values
    ):
        block_m_values = [16, block_n] if block_n != 16 else [block_n]
        for block_m in block_m_values:
            scalar_params = "i32,i32,i32,fp32,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32,i32,i32,i32"
            sig = f"*{dtype}:16,*{dtype}:16,*{dtype}:16,*{dtype}:16,*i32:16,*i32:16,{scalar_params},{block_m},{int(even_m)},{block_n},{int(even_n)},{block_d},{num_blocks_d}"
            prefix = "python compile.py sparse_attention_triton.py"
            filename = f"sparse_attention_kernel_{dtype}_m{block_m}_{int(even_m)}_n{block_n}_{int(even_n)}_d{block_d}_{num_blocks_d}_sm${{SM}}"
            name = f"sparse_attention_{dtype}_sm${{SM}}"
            num_warps = max(1, 2 ** int(math.log2(min(block_m, block_n, block_d) / 16)))
            num_stages = 2
            print(
                f'{prefix} -n block_sparse_attention_kernel -o {out_dir}/{filename} --out-name {name} -w {num_warps} -ns {num_stages} -s "{sig}" -g "(total_seq_len - past_seq_len + {block_m} - 1) / {block_m}, batch_size * num_heads, 1"'
            )

    print(f"cd {out_dir}")
    print(
        f"python ${{TRITON_ROOT}}/triton/tools/link.py sparse_attention_kernel_*.h -o sparse_attention_api_{dtype}_sm${{SM}}"
    )

    # Remove signature hash in code.
    suffix = "0d1d2d3d4d5d678910d11d12d13d14d15d16d17d18d19d20d21d22232425"
    print(f"for file in *.h; do sed -i 's/_{suffix}//g'  \"$file\"; done")
    print(f"for file in *.c; do sed -i 's/_{suffix}//g'  \"$file\"; done")

    # Keep the signature hash in kernel name.
    print(
        f"for file in *.c; do sed -i 's/block_sparse_attention_kernel/block_sparse_attention_kernel_{suffix}/g'  \"$file\"; done"
    )

    # Remove signature hash from filename since we use same signature for all kernels except constants.
    # and we have constants in filename so that we can distinguish them.
    print('for file in *.h; do mv -- "$file" "$(echo $file | cut -f 1 -d \'.\').h"; done')
    print('for file in *.c; do mv -- "$file" "$(echo $file | cut -f 1 -d \'.\').c"; done')

    filename = f"sparse_attention_api_{dtype}_sm${{SM}}"
    source1 = "CUstream stream, CUdeviceptr out, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, CUdeviceptr layout_csr_row_indices, CUdeviceptr layout_csr_col_indices, int32_t layout_csr_row_stride_h, int32_t layout_csr_col_stride_h, int32_t num_layout, float softmax_scale, int32_t stride_qb, int32_t stride_qh, int32_t stride_qm, int32_t stride_kb, int32_t stride_kh, int32_t stride_kn, int32_t stride_vb, int32_t stride_vh, int32_t stride_vn, int32_t stride_ob, int32_t stride_oh, int32_t stride_om, int32_t num_heads, int32_t num_kv_heads, int32_t total_seq_len, int32_t past_seq_len"
    target1 = "SparseAttentionParams& params"
    source2 = "stream, out, Q, K, V, layout_csr_row_indices, layout_csr_col_indices, layout_csr_row_stride_h, layout_csr_col_stride_h, num_layout, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_ob, stride_oh, stride_om, num_heads, num_kv_heads, total_seq_len, past_seq_len"
    target2 = "params"
    print(
        f"python -c \"import sys;lines=sys.stdin.read();lines=lines.replace('{source1}', '{target1}');lines=lines.replace('{source2}', '{target2}');print(lines)\" < \"{filename}.c\" > \"{filename}.cc\""
    )
    print(f"sed -i 's/CUresult/Status/g'  \"{filename}.cc\"")
    print(f"sed -i '/if /d'  \"{filename}.cc\"")
    print(f"sed -i '/CUDA_ERROR_INVALID_VALUE/d'  \"{filename}.cc\"")
    print(f"rm {filename}.c")
    print(f"rm {filename}.h")
    print(f"rm sparse_attention_kernel_{dtype}*.h")

    # rename *.c to *.cc
    print('for file in *.c; do mv -- "$file" "${file%.c}.cc"; done')

    # move kernel files to parent directory to update the files in repository.
    print("mv sparse_attention_kernel_* ../")

    print(
        f"echo please manually update sparse_attention_api.cc using content of sparse_attention_api_{dtype}_sm${{SM}}.cc, then remove sparse_attention_api_{dtype}_sm${{SM}}.cc"
    )


if __name__ == "__main__":
    generate_triton_compile_shell_script("fp16")