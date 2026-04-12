from numpy import dtype

import torch

B = 8
N = 32
N_H = 16
D = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def regular_attention(Q, K, V):
    QK_T = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
    print(QK_T.shape)
    print(Q.shape, K.shape, K.transpose(-2, -1).shape)
    A = torch.softmax(QK_T, dim=-1)
    return torch.matmul(A, V)



if __name__ == "__main__":
    # Create Q, K, V in proper multi-head shape: (B, num_heads, seq_len, head_dim)
    Q = torch.randn(B, N_H, N, D, device=device, dtype=torch.float16)
    K = torch.randn(B, N_H, N, D, device=device, dtype=torch.float16)
    V = torch.randn(B, N_H, N, D, device=device, dtype=torch.float16)

    # === Warmup (important for accurate timing) ===
    for _ in range(20):
        _ = regular_attention(Q, K, V)

    # === Proper Timing ===
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    output = regular_attention(Q, K, V)

    end.record()
    torch.cuda.synchronize()

    print(f"Regular attention time: {start.elapsed_time(end):.3f} ms")
    print(f"Output shape: {output.shape}")           # Should be [8, 16, 32, 64]
    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
