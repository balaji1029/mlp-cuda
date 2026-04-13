#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>

#define BLOCK_SIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void matmul(const float* A, const float* B, float* C, const size_t M, const size_t N, const size_t K, bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float value = 0;
        for (int i = 0; i < N; i++) {
            float a_val = transA ? A[i * M + row] : A[row * N + i];
            float b_val = transB ? B[col * N + i] : B[i * K + col];
            value += a_val * b_val;
        }
        C[row * K + col] = value;
    }
}

__global__ void matmul_bias(const float* A, const float* B, const float* bias, float* C, const size_t M, const size_t N, const size_t K, bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float value = bias[col];
        for (int i = 0; i < N; i++) {
            float a_val = transA ? A[i * M + row] : A[row * N + i];
            float b_val = transB ? B[col * N + i] : B[i * K + col];
            value += a_val * b_val;
        }
        C[row * K + col] = value;
    }
}

__global__ void relu(const float* A, float* C, const size_t M, const size_t N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        C[row * N + col] = fmaxf(0.0f, A[row * N + col]);
    }
}

__global__ void relu_backward(const float* dA, const float* Z, float* dZ, const size_t M, const size_t N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        dZ[row * N + col] = Z[row * N + col] > 0.0f ? dA[row * N + col] : 0.0f;
    }
}

__global__ void subtract(const float* A, const float* B, float* C, const size_t M, const size_t N, float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        C[row * N + col] = (A[row * N + col] - B[row * N + col]) * alpha;
    }
}

__global__ void bias_grad(const float* dZ, float* db, const size_t M, const size_t N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += dZ[i * N + col];
        }
        db[col] = sum;
    }
}


// Wrappers

void matmul_wrapper(const float* A, const float* B, float* C, const size_t M, const size_t N, const size_t K, bool transA = false, bool transB = false, cudaStream_t stream = 0) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(K, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
    matmul<<<gridSize, blockSize, 0, stream>>>(A, B, C, M, N, K, transA, transB);
}

void matmul_bias_wrapper(const float* A, const float* B, const float* bias, float* C, const size_t M, const size_t N, const size_t K, bool transA = false, bool transB = false, cudaStream_t stream = 0) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(K, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
    matmul_bias<<<gridSize, blockSize, 0, stream>>>(A, B, bias, C, M, N, K, transA, transB);
}

void relu_wrapper(const float* A, float* C, const size_t M, const size_t N, cudaStream_t stream = 0) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
    relu<<<gridSize, blockSize, 0, stream>>>(A, C, M, N);
}

void relu_backward_wrapper(const float* dA, const float* Z, float* dZ, const size_t M, const size_t N, cudaStream_t stream = 0) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
    relu_backward<<<gridSize, blockSize, 0, stream>>>(dA, Z, dZ, M, N);
}

void subtract_wrapper(const float* A, const float* B, float* C, const size_t M, const size_t N, float alpha, cudaStream_t stream = 0) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));
    subtract<<<gridSize, blockSize, 0, stream>>>(A, B, C, M, N, alpha);
}

void bias_grad_wrapper(const float* dZ, float* db, const size_t M, const size_t N, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = CEIL_DIV(N, threads);
    bias_grad<<<blocks, threads, 0, stream>>>(dZ, db, M, N);
}

#define BATCH 1024
#define IN 512
#define H1 2048
#define H2 2048
#define H3 2048
#define OUT 512

int main(int argc, char* argv[]) {
    srand(42);

    // Host arrays for initialization
    // float* h_W1 = (float*)malloc(IN * H1 * sizeof(float));
    // float* h_W2 = (float*)malloc(H1 * H2 * sizeof(float));
    // float* h_W3 = (float*)malloc(H2 * H3 * sizeof(float));
    // float* h_W4 = (float*)malloc(H3 * OUT * sizeof(float));
    // float* h_X  = (float*)malloc(BATCH * IN * sizeof(float));
    // float* h_Y  = (float*)malloc(BATCH * OUT * sizeof(float));

    // // He initialization for weights
    // float scale1 = sqrtf(2.0f / IN);
    // for (int i = 0; i < IN * H1; i++) h_W1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale1;
    // float scale2 = sqrtf(2.0f / H1);
    // for (int i = 0; i < H1 * H2; i++) h_W2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale2;
    // float scale3 = sqrtf(2.0f / H2);
    // for (int i = 0; i < H2 * H3; i++) h_W3[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale3;
    // float scale4 = sqrtf(2.0f / H3);
    // for (int i = 0; i < H3 * OUT; i++) h_W4[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale4;

    // // Random input and target data
    // for (int i = 0; i < BATCH * IN; i++)  h_X[i] = (float)rand() / RAND_MAX;
    // for (int i = 0; i < BATCH * OUT; i++) h_Y[i] = (float)rand() / RAND_MAX;

    // Device allocations - weights
    float *W1, *W2, *W3, *W4;
    cudaMalloc((void**)&W1, IN * H1 * sizeof(float));
    cudaMalloc((void**)&W2, H1 * H2 * sizeof(float));
    cudaMalloc((void**)&W3, H2 * H3 * sizeof(float));
    cudaMalloc((void**)&W4, H3 * OUT * sizeof(float));

    // Device allocations - biases (zero-initialized)
    float *b1, *b2, *b3, *b4;
    cudaMalloc((void**)&b1, H1 * sizeof(float));
    cudaMalloc((void**)&b2, H2 * sizeof(float));
    cudaMalloc((void**)&b3, H3 * sizeof(float));
    cudaMalloc((void**)&b4, OUT * sizeof(float));
    cudaMemset(b1, 0, H1 * sizeof(float));
    cudaMemset(b2, 0, H2 * sizeof(float));
    cudaMemset(b3, 0, H3 * sizeof(float));
    cudaMemset(b4, 0, OUT * sizeof(float));

    // Device allocations - input/target
    float *X, *Y;
    cudaMalloc((void**)&X, BATCH * IN * sizeof(float));
    cudaMalloc((void**)&Y, BATCH * OUT * sizeof(float));

    // Device allocations - forward pass activations
    float *Z1, *Z2, *Z3, *Z4;
    float *A1, *A2, *A3;
    cudaMalloc((void**)&Z1, BATCH * H1 * sizeof(float));
    cudaMalloc((void**)&Z2, BATCH * H2 * sizeof(float));
    cudaMalloc((void**)&Z3, BATCH * H3 * sizeof(float));
    cudaMalloc((void**)&Z4, BATCH * OUT * sizeof(float));
    cudaMalloc((void**)&A1, BATCH * H1 * sizeof(float));
    cudaMalloc((void**)&A2, BATCH * H2 * sizeof(float));
    cudaMalloc((void**)&A3, BATCH * H3 * sizeof(float));

    // Device allocations - weight gradients
    float *dW1, *dW2, *dW3, *dW4;
    cudaMalloc((void**)&dW1, IN * H1 * sizeof(float));
    cudaMalloc((void**)&dW2, H1 * H2 * sizeof(float));
    cudaMalloc((void**)&dW3, H2 * H3 * sizeof(float));
    cudaMalloc((void**)&dW4, H3 * OUT * sizeof(float));

    // Device allocations - bias gradients
    float *db1, *db2, *db3, *db4;
    cudaMalloc((void**)&db1, H1 * sizeof(float));
    cudaMalloc((void**)&db2, H2 * sizeof(float));
    cudaMalloc((void**)&db3, H3 * sizeof(float));
    cudaMalloc((void**)&db4, OUT * sizeof(float));

    // Device allocations - backward pass intermediates
    float *dZ4, *dZ3, *dZ2, *dZ1;
    float *dA3, *dA2, *dA1;
    cudaMalloc((void**)&dZ4, BATCH * OUT * sizeof(float));
    cudaMalloc((void**)&dZ3, BATCH * H3 * sizeof(float));
    cudaMalloc((void**)&dZ2, BATCH * H2 * sizeof(float));
    cudaMalloc((void**)&dZ1, BATCH * H1 * sizeof(float));
    cudaMalloc((void**)&dA3, BATCH * H3 * sizeof(float));
    cudaMalloc((void**)&dA2, BATCH * H2 * sizeof(float));
    cudaMalloc((void**)&dA1, BATCH * H1 * sizeof(float));

    // Copy data to device
    // cudaMemcpy(W1, h_W1, IN * H1 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(W2, h_W2, H1 * H2 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(W3, h_W3, H2 * H3 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(W4, h_W4, H3 * OUT * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(X, h_X, BATCH * IN * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(Y, h_Y, BATCH * OUT * sizeof(float), cudaMemcpyHostToDevice);

    // Streams: compute_stream for gradient propagation, grad_stream for dW/db
    cudaStream_t compute_stream, grad_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&grad_stream);

    // Events for synchronization between streams
    cudaEvent_t start, end;
    cudaEvent_t dz_ready, grads_done;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&dz_ready);
    cudaEventCreate(&grads_done);

    cudaEventRecord(start, compute_stream);

    // === Forward pass (sequential on compute_stream) ===
    matmul_bias_wrapper(X, W1, b1, Z1, BATCH, IN, H1, false, false, compute_stream);
    relu_wrapper(Z1, A1, BATCH, H1, compute_stream);
    matmul_bias_wrapper(A1, W2, b2, Z2, BATCH, H1, H2, false, false, compute_stream);
    relu_wrapper(Z2, A2, BATCH, H2, compute_stream);
    matmul_bias_wrapper(A2, W3, b3, Z3, BATCH, H2, H3, false, false, compute_stream);
    relu_wrapper(Z3, A3, BATCH, H3, compute_stream);
    matmul_bias_wrapper(A3, W4, b4, Z4, BATCH, H3, OUT, false, false, compute_stream);

    // === Backward pass (2 streams) ===
    // dZ4 = 2/(BATCH*OUT) * (Z4 - Y)
    subtract_wrapper(Z4, Y, dZ4, BATCH, OUT, 2.0f / (BATCH * OUT), compute_stream);

    // Signal that dZ4 is ready
    cudaEventRecord(dz_ready, compute_stream);
    cudaStreamWaitEvent(grad_stream, dz_ready);

    // grad_stream: Layer 4 weight/bias gradients (reads A3, dZ4)
    matmul_wrapper(A3, dZ4, dW4, H3, BATCH, OUT, true, false, grad_stream);
    bias_grad_wrapper(dZ4, db4, BATCH, OUT, grad_stream);

    // compute_stream: propagate to layer 3 (writes dA3, dZ3 — separate buffers, no conflict)
    matmul_wrapper(dZ4, W4, dA3, BATCH, OUT, H3, false, true, compute_stream);
    relu_backward_wrapper(dA3, Z3, dZ3, BATCH, H3, compute_stream);

    // Signal that dZ3 is ready
    cudaEventRecord(dz_ready, compute_stream);
    cudaStreamWaitEvent(grad_stream, dz_ready);

    // grad_stream: Layer 3 weight/bias gradients (reads A2, dZ3)
    matmul_wrapper(A2, dZ3, dW3, H2, BATCH, H3, true, false, grad_stream);
    bias_grad_wrapper(dZ3, db3, BATCH, H3, grad_stream);

    // compute_stream: propagate to layer 2
    matmul_wrapper(dZ3, W3, dA2, BATCH, H3, H2, false, true, compute_stream);
    relu_backward_wrapper(dA2, Z2, dZ2, BATCH, H2, compute_stream);

    // Signal that dZ2 is ready
    cudaEventRecord(dz_ready, compute_stream);
    cudaStreamWaitEvent(grad_stream, dz_ready);

    // grad_stream: Layer 2 weight/bias gradients (reads A1, dZ2)
    matmul_wrapper(A1, dZ2, dW2, H1, BATCH, H2, true, false, grad_stream);
    bias_grad_wrapper(dZ2, db2, BATCH, H2, grad_stream);

    // compute_stream: propagate to layer 1
    matmul_wrapper(dZ2, W2, dA1, BATCH, H2, H1, false, true, compute_stream);
    relu_backward_wrapper(dA1, Z1, dZ1, BATCH, H1, compute_stream);

    // Signal that dZ1 is ready
    cudaEventRecord(dz_ready, compute_stream);
    cudaStreamWaitEvent(grad_stream, dz_ready);

    // grad_stream: Layer 1 weight/bias gradients (reads X, dZ1)
    matmul_wrapper(X, dZ1, dW1, IN, BATCH, H1, true, false, grad_stream);
    bias_grad_wrapper(dZ1, db1, BATCH, H1, grad_stream);

    // Wait for both streams to finish
    cudaEventRecord(end, compute_stream);
    cudaEventRecord(grads_done, grad_stream);
    cudaEventSynchronize(end);
    cudaEventSynchronize(grads_done);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("%f\n", elapsedTime);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaEventDestroy(dz_ready);
    cudaEventDestroy(grads_done);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(grad_stream);

    size_t freeMem, totalMem;

    cudaMemGetInfo(&freeMem, &totalMem);

    std::cout << "Free memory: " << freeMem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total memory: " << totalMem / (1024.0 * 1024.0) << " MB" << std::endl;

    // free(h_W1); free(h_W2); free(h_W3); free(h_W4);
    // free(h_X); free(h_Y);

    cudaFree(W1); cudaFree(W2); cudaFree(W3); cudaFree(W4);
    cudaFree(b1); cudaFree(b2); cudaFree(b3); cudaFree(b4);
    cudaFree(X); cudaFree(Y);
    cudaFree(Z1); cudaFree(Z2); cudaFree(Z3); cudaFree(Z4);
    cudaFree(A1); cudaFree(A2); cudaFree(A3);
    cudaFree(dW1); cudaFree(dW2); cudaFree(dW3); cudaFree(dW4);
    cudaFree(db1); cudaFree(db2); cudaFree(db3); cudaFree(db4);
    cudaFree(dZ4); cudaFree(dZ3); cudaFree(dZ2); cudaFree(dZ1);
    cudaFree(dA3); cudaFree(dA2); cudaFree(dA1);

    return 0;
}