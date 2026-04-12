#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void flash_attention_kernel(
    float* Q, float* K, float* V, float* O, float* l, float* m, int B, int H, int N, int D, int Tr, int Tc, int Br, int Bc
) {
    int b = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index

    int tx = threadIdx.x; // Thread index within block

    int qkv_base = (b * H + h) * N * D; // Base index for Q, K, V
    int lm_base = (b * H + h) * N; // Base index for l and m

    __shared__ float smem[]; // Shared memory for Q, K, V, and output tile
    float* Qi = smem; // Q tile: Br x D
    float* Kj = Qi + Br * D; // K tile: Bc x D
    float* Vj = Kj + Bc * D; // V tile: Bc x D
    float* S = Vj + Bc * D; // Output tile: Br x Bc

    for (int j = 0; j < Tc; j++) {

        for (int x = 0; x < D; x++) {
            int kv_idx = qkv_base + (j * Bc + tx) * D + x;

            Kj[tx * D + x] = K[kv_idx];
            Vj[tx * D + x] = V[kv_idx];
        }

        __syncthreads();

        for (int i = 0; i < Tr; i++) {
            for (int x = 0; x < D; x++) {
                int q_idx = qkv_base + (i * Br + tx) * D + x;

                Qi[tx * D + x] = Q[q_idx];
            }

            int lm_idx = lm_base + i * Br + tx;

            float l_prev = l[lm_idx];
            float m_prev = m[lm_idx];

            float row_m = -INFINITY;

            for (int x = 0; x < Bc; x++) {
                float prod = 0.0f;
                for (int y = 0; y < D; y++) {
                    prod += Qi[tx * D + y] * Kj[x * D + y];
                }
                row_m = max(row_m, S[tx * Bc + x] = prod / sqrtf(D));
            }

            float l_ij = 0.0f;
            for (int x = 0; x < Bc; x++) {
                l_ij += S[tx * Bc + x] = expf(S[tx * Bc + x] - row_m);
            }

            float mi_new = max(m_prev, row_m);
            float li_new = l_prev * expf(m_prev - mi_new) + l_ij * expf(row_m - mi_new);

            for (int x = 0; x < D; x++) {
                float pv = 0.0f;
                for (int y = 0; y < Bc; y++)
                    pv += S[tx * Bc + y] * Vj[y * D + x];
                int o_idx = qkv_base + (i * Br + tx) * D + x;
                O[o_idx] = (l_prev * expf(m_prev - mi_new) * O[o_idx] + expf(row_m - mi_new) * pv) / li_new;
            }
            l[lm_idx] = li_new;
            m[lm_idx] = mi_new;
        }
        __syncthreads();
    }


}

int main() {
    // Example dimensions
    int B = 8; // Batch size
    int H = 16; // Number of heads
    int N = 32; // Sequence length
    int D = 64; // Head dimension

    // Allocate memory for Q, K, V, O, l, m on device
    float* Q, * K, * V, * O, * l, * m;
    cudaMalloc(&Q, B * H * N * D * sizeof(float));
    cudaMalloc(&K, B * H * N * D * sizeof(float));
    cudaMalloc(&V, B * H * N * D * sizeof(float));
    cudaMalloc(&O, B * H * N * D * sizeof(float));
    cudaMalloc(&l, B * H * N * sizeof(float));
    cudaMalloc(&m, B * H * N * sizeof(float));

    int Br = 32;
    int Bc = 32;

    int Tr = CEIL_DIV(N, Br);
    int Tc = CEIL_DIV(N, Bc);

    // Launch the flash attention kernel
    dim3 grid(B, H);
    dim3 block(Br);
    flash_attention_kernel << <grid, block >> > (Q, K, V, O, l, m, B, H, N, D, Tr, Tc, Br, Bc);

    // Free device memory
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(O);
    cudaFree(l);
    cudaFree(m);

    return 0;

}