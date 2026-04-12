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
    Q = torch.randn(B, N, D, device=device, dtype=torch.float16)
    K = torch.randn(B, N, D, device=device, dtype=torch.float16)
    V = torch.randn(B, N, D, device=device, dtype=torch.float16)

    time = torch.cuda.Event(enable_timing=True)
    time.record()
    output = regular_attention(Q, K, V)
    time.record()
    torch.cuda.synchronize()


    print(f"Regular attention time: {time.elapsed_time()} ms")
    print(output.shape)
