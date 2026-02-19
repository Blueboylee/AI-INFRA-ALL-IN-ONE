/**
 * PMPP Chapter 3 - Matrix Multiplication (Basic)
 *
 * 基础矩阵乘法: C = A × B
 * A: M×K, B: K×N, C: M×N
 *
 * 每个线程计算 C 的一个元素: C[row][col] = Σ A[row][k] * B[k][col]
 * 使用 2D Grid/Block, 演示行优先内存布局与多维索引映射。
 *
 * 注: 这是未优化的基础版本 (无 Shared Memory Tiling),
 *     优化版本将在 Chapter 5 实现。
 *
 * 编译: nvcc -o ch03_matmul ch03_matmul.cu
 * 运行: ./ch03_matmul [M] [K] [N]
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// CPU 基线: 朴素三重循环
// ============================================================

void matmul_cpu(const float* A, const float* B, float* C,
                int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// ============================================================
// CUDA Kernel: 基础矩阵乘法 (每线程计算 C 的一个元素)
// ============================================================

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================
// Host 端流程
// ============================================================

void matmul_gpu(const float* h_A, const float* h_B, float* h_C,
                int M, int K, int N) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    printf("  Grid: (%d, %d), Block: (%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
    printf("  GPU Kernel time: %.3f ms (%.1f GFLOPS)\n", ms, gflops);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int K = (argc > 2) ? atoi(argv[2]) : 1024;
    int N = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("=== PMPP Ch03: Matrix Multiplication (Basic) ===\n");
    printf("A: %dx%d, B: %dx%d, C: %dx%d\n\n", M, K, K, N, M, N);

    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C_cpu = (float*)malloc(M * N * sizeof(float));
    float* h_C_gpu = (float*)malloc(M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    // CPU
    double t0 = get_time_ms();
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    double t1 = get_time_ms();
    double cpu_gflops = 2.0 * M * N * K / ((t1 - t0) / 1000.0) / 1e9;
    printf("[CPU] Time: %.1f ms (%.2f GFLOPS)\n", t1 - t0, cpu_gflops);

    // GPU
    printf("[GPU]\n");
    matmul_gpu(h_A, h_B, h_C_gpu, M, K, N);

    // 验证
    int mismatch = 0;
    float max_err = 0;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_C_cpu[i] - h_C_gpu[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-2f) mismatch++;
    }
    printf("\n[Verify] Max error: %.6f, Mismatches: %d / %d\n",
           max_err, mismatch, M * N);
    printf("%s\n", mismatch == 0 ? "[PASS]" : "[FAIL]");

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    return 0;
}
