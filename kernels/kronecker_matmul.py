import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=3, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def kronecker_matmul_kernel(
    x_ptr: tl.tensor,
    u_ptr: tl.tensor,
    v_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    M_np2: tl.constexpr,
    N_np2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Matmul between X and U \\otimes V.

    Args:
        x_ptr: pointer to X of shape (B, L, M, N)
        u_ptr: pointer to U of shape (M, M)
        v_ptr: pointer to V of shape (N, N)
        out_ptr: pointer to output of shape (B, M, N)
    """
    pid = tl.program_id(axis=0) * tl.num_programs(axis=1) + tl.program_id(axis=1)
    pid_mn = tl.program_id(axis=2)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_xm = tl.arange(0, M_np2)
    offs_xn = tl.arange(0, N_np2)
    offs_um = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_vn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    x_ptrs = x_ptr + pid * (M * N) + offs_xm[:, None] * N + offs_xn[None, :]
    u_ptrs = u_ptr + offs_um[:, None] * M + offs_xm[None, :]
    v_ptrs = v_ptr + offs_xn[:, None] * N + offs_vn[None, :]

    x = tl.load(x_ptrs, mask=(offs_xm < M)[:, None] * (offs_xn < N)[None, :])
    u = tl.load(u_ptrs, mask=(offs_um < M)[:, None] * (offs_xm < M)[None, :])
    v = tl.load(v_ptrs, mask=(offs_xn < N)[:, None] * (offs_vn < N)[None, :])

    dtype = x.dtype

    out = tl.dot(tl.dot(u, x).to(dtype), v).to(dtype)

    out_ptrs = out_ptr + pid * (M * N) + offs_um[:, None] * N + offs_vn[None, :]
    tl.store(out_ptrs, out, mask=(offs_um < M)[:, None] * (offs_vn < N)[None, :])


def kronecker_matmul(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, sequence_length: int):
    """
    Matmul between X and U \\otimes V.

    Args:
        x: tensor of shape (B, M, N)
        u: tensor of shape (M, M)
        v: tensor of shape (N, N)
        sequence_length: number of tokens in the sequence

    Returns:
        out: tensor of shape (B, M, N)
    """
    # Check dimensions
    assert x.ndim == 3
    assert u.ndim == 2
    assert v.ndim == 2
    B, M, N = x.shape
    assert u.shape[0] == u.shape[1] == M
    assert v.shape[0] == v.shape[1] == N

    out = torch.empty_like(x)

    grid = lambda meta: (
        B // sequence_length,
        sequence_length,
        math.ceil(M / meta["BLOCK_SIZE_M"]) * math.ceil(N / meta["BLOCK_SIZE_N"]),
    )

    kronecker_matmul_kernel[grid](
        x,
        u,
        v,
        out,
        M,
        N,
        triton.next_power_of_2(M),
        triton.next_power_of_2(N),
    )
    return out
