#include <cuda_runtime.h>
#include <random>
#include <iostream>

#define BLOCK_SIZE 32
#define CEIL_DIV(a, b) (a + b - 1) / b

__global__ void transpose(const float* A, float* B, const size_t M, const size_t N) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < M && y < N)
        B[y * M + x] = A[x * N + y];
}

int main(int argc, char* argv[]) {
    float* A, * B;
    int M = 1024;
    int N = 1024;
    if (argc == 2) {
        N = atoi(argv[1]);
    }
    if (argc == 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }


    A = new float[M * N];
    B = new float[N * M];

    for (int i = 0; i < M * N; i++) {
        A[i] = ((float)rand()) / RAND_MAX;
    }

    float* dev_A, * dev_B;

    cudaMalloc(&dev_A, M * N * sizeof(float));
    cudaMalloc(&dev_B, M * N * sizeof(float));

    cudaMemcpy(dev_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    cudaStream_t stream;
    cudaEvent_t start, stop;

    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    transpose << < gridSize, blockSize, 0, stream >> > (dev_A, dev_B, M, N);
    cudaEventRecord(stop, stream);
    cudaMemcpyAsync(B, dev_B, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize();
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i * N + j] != B[j * M + i]) {
                std::cout << "Wrong output :(" << std::endl;
                goto end;
            }
        }
    }
    end:
    std::cout << elapsedTime << std::endl;

}