from numpy import dtype

import torch

B = 8
N = 32
N_H = 16
D = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def regular_attention(Q, K, V):
    QK_T = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
    # print(QK_T.shape)
    # print(Q.shape, K.shape, K.transpose(-2, -1).shape)
    A = torch.softmax(QK_T, dim=-1)
    return torch.matmul(A, V)

def flash_attention_naive(Q, K, V):
    M = 32 * 1024
    Bc = M // (4 * D)
    Br = min(Bc, D)
    Tc = (N + Bc - 1) // Bc
    Tr = (N + Br - 1) // Br

    O = torch.zeros_like(Q, device=device, dtype=torch.float16)
    l = torch.zeros((B, N_H, N, 1), device=device, dtype=torch.float16)
    m = torch.zeros((B, N_H, N, 1), device=device, dtype=torch.float16)

    for j in range(Tc):
        K_j = K[:, :, j*Bc:(j+1)*Bc, :]
        V_j = V[:, :, j*Bc:(j+1)*Bc, :]
        for i in range(Tr):
            Q_i = Q[:, :, i*Br:(i+1)*Br, :]
            S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) / (D ** 0.5)
            m_ij = torch.max(S_ij, dim=-1, keepdim=True).values
            l_ij = torch.sum(torch.exp(S_ij - m_ij), dim=-1, keepdim=True)
            O[:, :, i*Br:(i+1)*Br, :] += torch.matmul(torch.exp(S_ij - m_ij) / l_ij, V_j)
            m[:, :, i*Br:(i+1)*Br, :] = torch.max(m[:, :, i*Br:(i+1)*Br, :], m_ij)
            l[:, :, i*Br:(i+1)*Br, :] = l[:, :, i*Br:(i+1)*Br, :] * torch.exp(m[:, :, i*Br:(i+1)*Br, :] - m_ij) + l_ij * torch.exp(m[:, :, i*Br:(i+1)*Br, :] - m_ij)
    return O

if __name__ == "__main__":
    for N in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
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

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output_flash = flash_attention_naive(Q, K, V)
        end.record()

        torch.cuda.synchronize()

        if not torch.allclose(output, output_flash, atol=1e-3):
            print("Outputs do not match!")

        print(f"Flash attention time: {start.elapsed_time(end):.3f} ms")

    # print(f"Output shape: {output.shape}")           # Should be [8, 16, 32, 64]
    # print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
