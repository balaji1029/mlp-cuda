#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <chrono>

#define NELEM 6
#define BLOCK_SIZE 32
#define CEIL_DIV(a, b) (a + b - 1) / b

#define CHECK_CUDA(x) \
{ \
    cudaError_t err = x; \
    if(err != cudaSuccess){ \
        std::cerr << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

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

        for (int k = 0; k < NELEM; k++) {
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

int main(int argc, char** argv) {
    size_t N = 1024, B = 32;
    if (argc == 2) {
        N = std::stoul(argv[1]);
    }

    float* W1 = new float[N * N * sizeof(float)];
    float* W2 = new float[(N * N * sizeof(float))];

    float* input[4];
    for (int i = 0; i < 4; i++) {
        input[i] = new float[(N * (B / 4) * sizeof(float))];
    }

    for (int i = 0; i < N * N; i++) {
        W1[i] = static_cast<float>(rand()) / RAND_MAX;
        W2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < N * (B / 4); i++) {
        for (int j = 0; j < 4; j++) {
            input[j][i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float* d_W1, * d_W2;

    float* d_input[4];
    float* d_output1[4];
    float* d_output2[4];

    cudaMalloc(&d_W1, N * N * sizeof(float));
    cudaMalloc(&d_W2, N * N * sizeof(float));

    cudaMemcpy(d_input, input, N * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize1(CEIL_DIV(B, BLOCK_SIZE), CEIL_DIV(N, NELEM * BLOCK_SIZE));
    dim3 blockSize1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize_relu(CEIL_DIV(B, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    dim3 blockSize_relu(BLOCK_SIZE, BLOCK_SIZE);

    cudaStream_t stream[4];
    cudaEvent_t start[4], end[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&start[i]);
        cudaEventCreate(&end[i]);
    }
    auto start_chrono = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; i++) {
        cudaEventRecord(start[i], stream[i]);
        cudaMemcpyAsync(d_input[i], input[i], N * (B / 4) * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        tiling_matmul << < gridSize1, blockSize1, 0, stream[i] >> > (d_W1, d_input[i], d_output1[i], N, (B / 4), N);
        relu_kernel << < gridSize_relu, blockSize_relu, 0, stream[i] >> > (d_output1[i], N, (B / 4));
        tiling_matmul << < gridSize1, blockSize1, 0, stream[i] >> > (d_W2, d_output1[i], d_output2[i], N, (B / 4), N);
        cudaEventRecord(end[i], stream[i]);
    }
    for (int i = 0; i < 4; i++) {
        cudaStreamSynchronize(stream[i]);
    }
    cudaDeviceSynchronize();
    auto end_chrono = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = end_chrono - start_chrono;
    std::cout << elapsedTime.count() << std::endl;
    for (int i = 0; i < 4; i++) {
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start[i], end[i]);
        // std::cout << "Stream " << i << ": " << elapsedTime << " ms" << std::endl;
    }

    delete[] W1;
    delete[] W2;
    for (int i = 0; i < 4; i++) {
        delete[] input[i];
    }

    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);

    for (int i = 0; i < 4; i++) {
        cudaEventDestroy(start[i]);
        cudaEventDestroy(end[i]);
    }
    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(stream[i]);
    }

    return 0;

}