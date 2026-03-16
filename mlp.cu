#include <cuda_runtime.h>
#include <string>
#include <iostream>

#define NELEM 6
#define BLOCK_SIZE 32
#define CEIL_DIV(a, b) (a + b - 1) / b

__global__ void relu_kernel(float* data, size_t M, size_t N) {
    int globalX = blockDim.x * blockIdx.x + threadIdx.x;
    int globalY = blockDim.y * blockIdx.y + threadIdx.y;

    if (globalX < N && globalY < M) {
        size_t idx = globalY * N + globalX;
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void tiling_matmul(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    __shared__ float tileA[NELEM * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int localX = threadIdx.x;
    int localY = threadIdx.y;

    int globalX = blockDim.x * blockIdx.x + localX;
    int globalY = blockDim.y * (blockIdx.y * NELEM) + localY;

    float ans[NELEM] = { 0.0f };

    int numTiles = CEIL_DIV(K, BLOCK_SIZE);

    for (int i = 0; i < numTiles; i++) {
        int tileAx = i * BLOCK_SIZE + localX;
        int tileAy = globalY;

        for (int k=0; k < NELEM; k++) {
            tileA[localY + k * BLOCK_SIZE][localX] = ((tileAy + k * BLOCK_SIZE) < M && tileAx < K) ? A[(tileAy + k * BLOCK_SIZE) * K + tileAx] : 0.0f;
        }

        int tileBx = globalX;
        int tileBy = i * BLOCK_SIZE + localY;

        tileB[localY][localX] = (tileBy < K && tileBx < N) ? B[tileBy * N + tileBx] : 0.0f;

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            for (int k = 0; k < NELEM; k++) {
                ans[k] += tileA[localY + k * BLOCK_SIZE][j] * tileB[j][localX];
            }
        }

        __syncthreads();
    }

    for (int k = 0; k < NELEM; k++)
        if ((globalY + k * BLOCK_SIZE) < M && globalX < N)
            C[(globalY + k * BLOCK_SIZE) * N + globalX] = ans[k];
}

int main(int argc, char**argv) {
    size_t N = 1024, B = 32;
    if (argc == 2) {
        N = std::stoul(argv[1]);
    }

    float* W1 = (float*)malloc(N * N * sizeof(float));
    float* W2 = (float*)malloc(N * N * sizeof(float));

    float* input = (float*)malloc(N * B * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        W1[i] = static_cast<float>(rand()) / RAND_MAX;
        W2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < N * B; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* d_W1, * d_W2, * d_input, * d_output1, * d_output2;

    cudaMalloc(&d_input, N * B * sizeof(float));
    cudaMalloc(&d_W1, N * N * sizeof(float));
    cudaMalloc(&d_output1, N * B * sizeof(float));
    cudaMalloc(&d_output2, N * B * sizeof(float));

    cudaMemcpy(d_input, input, N * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize1(CEIL_DIV(B, BLOCK_SIZE), CEIL_DIV(N, NELEM * BLOCK_SIZE));
    dim3 blockSize1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize_relu(CEIL_DIV(B, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    dim3 blockSize_relu(BLOCK_SIZE, BLOCK_SIZE);
    
    cudaStream_t stream;
    cudaEvent_t start, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);

    tiling_matmul <<< gridSize1, blockSize1, 0, stream >>> (d_W1, d_input, d_output1, N, B, N);
    relu_kernel <<< gridSize_relu, blockSize_relu, 0, stream >>> (d_output1, N, B);
    tiling_matmul <<< gridSize1, blockSize1, 0, stream >>> (d_W2, d_output1, d_output2, N, B, N);

    cudaEventRecord(end, stream);
    cudaStreamSynchronize(stream);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    
    std::cout << elapsedTime << std::endl;

    // cudaMalloc(&d_W2, N * N * sizeof(float));
    // cudaMalloc(&d_output2, N * B * sizeof(float));

    delete [] W1;
    delete [] W2;
    delete [] input;

    cudaFree(d_W1);
    cudaFree(d_input);
    cudaFree(d_output1);

    return 0;

}