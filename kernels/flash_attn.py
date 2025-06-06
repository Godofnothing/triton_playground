import math
import torch
import triton
import triton.language as tl


@triton.jit
def tl_exp(x: tl.tensor):
    log2_e = 1.44269504
    return tl.exp2(log2_e * x)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_R": 32, "BLOCK_SIZE_C": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE_R": 32, "BLOCK_SIZE_C": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_R": 32, "BLOCK_SIZE_C": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_R": 64, "BLOCK_SIZE_C": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE_R": 64, "BLOCK_SIZE_C": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_R": 64, "BLOCK_SIZE_C": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_R": 128, "BLOCK_SIZE_C": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE_R": 128, "BLOCK_SIZE_C": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_R": 128, "BLOCK_SIZE_C": 128}, num_stages=4, num_warps=4),
    ],
    key=["N"],
)
@triton.jit
def flash_attn_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    o_ptr: tl.tensor,
    scale: float,
    N: int,
    D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Define constants and offsets
    pid_batch = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_r = pid_m * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_d = tl.arange(0, D)
   
    # Initialize accumulators
    m_i = tl.full((BLOCK_SIZE_R,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_R,), tl.float32)
    acc = tl.zeros((BLOCK_SIZE_R, D), tl.float32)

    # Load the Q block for this program. It remains constant throughout the loop.
    q_offs = pid_batch * N * D + offs_r[:, None] * D + offs_d[None, :]
    q_mask = (offs_r < N)[:, None]
    q_i = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)

    dtype = q_i.dtype

    # Loop over column blocks of K and V.
    for j in range(0, tl.cdiv(N, BLOCK_SIZE_C)):
        offs_c = j * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
        # Load K and V blocks
        k_offs = pid_batch * N * D + offs_c[:, None] * D + offs_d[None, :]
        v_offs = pid_batch * N * D + offs_c[:, None] * D + offs_d[None, :]
        kv_mask = (offs_c < N)[:, None]
        k_j = tl.load(k_ptr + k_offs, mask=kv_mask, other=0.0)
        v_j = tl.load(v_ptr + v_offs, mask=kv_mask, other=0.0)

        # Compute Q K.T
        s_ij = scale * tl.dot(q_i, k_j.trans(1, 0)) # shape (BLOCK_SIZE_R, BLOCK_SIZE_C)

        m_ij = s_ij.max(axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Correctly rescale weights and accumulator
        p_ij = tl_exp(s_ij - m_new[:, None])
        alpha = tl_exp(m_i - m_new)

        l_ij = tl.sum(p_ij, axis=1)
        l_new = alpha * l_i + l_ij

        # Rescale old accumulator
        acc = acc * alpha[:, None]
        # Add new value
        acc += tl.dot(p_ij.to(dtype), v_j)

        # Update statistics
        m_i = m_new
        l_i = l_new

    # Divide accumulator by the final denominator
    acc = acc / l_i[:, None]

    tl.store(o_ptr + q_offs, acc.to(dtype), mask=q_mask)


def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    assert q.shape == k.shape == v.shape
    B, N, D = q.shape
    o = torch.zeros_like(q)

    grid = lambda meta: (B, tl.cdiv(N, meta["BLOCK_SIZE_R"]))

    flash_attn_kernel[grid](
        q,
        k,
        v,
        o,
        scale=1.0 / math.sqrt(D),
        N=N,
        D=D,
    )
    return o
