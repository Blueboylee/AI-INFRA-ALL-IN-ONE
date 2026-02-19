/**
 * PMPP Chapter 3 - Color to Grayscale Conversion
 *
 * 将 RGB 图像转换为灰度图, 演示 2D Grid/Block 配置与多维索引映射。
 *
 * 灰度公式 (ITU-R BT.601):
 *   gray = 0.21 * R + 0.72 * G + 0.07 * B
 *
 * 编译: nvcc -o ch03_color2gray ch03_color2gray.cu
 * 运行: ./ch03_color2gray [width] [height]
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
// CPU 基线
// ============================================================

void color2gray_cpu(const unsigned char* rgb, unsigned char* gray,
                    int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = row * width + col;
            unsigned char r = rgb[3 * idx + 0];
            unsigned char g = rgb[3 * idx + 1];
            unsigned char b = rgb[3 * idx + 2];
            gray[idx] = (unsigned char)(0.21f * r + 0.72f * g + 0.07f * b);
        }
    }
}

// ============================================================
// CUDA Kernel: 2D Grid, 每个线程处理一个像素
// ============================================================

__global__ void color2gray_kernel(const unsigned char* rgb,
                                   unsigned char* gray,
                                   int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        unsigned char r = rgb[3 * idx + 0];
        unsigned char g = rgb[3 * idx + 1];
        unsigned char b = rgb[3 * idx + 2];
        gray[idx] = (unsigned char)(0.21f * r + 0.72f * g + 0.07f * b);
    }
}

// ============================================================
// Host 端流程
// ============================================================

void color2gray_gpu(const unsigned char* h_rgb, unsigned char* h_gray,
                    int width, int height) {
    size_t rgb_size = 3 * width * height * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);

    unsigned char *d_rgb, *d_gray;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_size));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_size));

    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgb_size, cudaMemcpyHostToDevice));

    // 2D Block 和 Grid 配置
    dim3 blockDim(16, 16);   // 16×16 = 256 threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    color2gray_kernel<<<gridDim, blockDim>>>(d_rgb, d_gray, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
}

int main(int argc, char** argv) {
    int width  = (argc > 1) ? atoi(argv[1]) : 1920;
    int height = (argc > 2) ? atoi(argv[2]) : 1080;

    printf("=== PMPP Ch03: Color to Grayscale ===\n");
    printf("Image size: %d x %d\n\n", width, height);

    size_t rgb_size = 3 * width * height;
    size_t gray_size = width * height;

    unsigned char* h_rgb      = (unsigned char*)malloc(rgb_size);
    unsigned char* h_gray_cpu = (unsigned char*)malloc(gray_size);
    unsigned char* h_gray_gpu = (unsigned char*)malloc(gray_size);

    srand(42);
    for (size_t i = 0; i < rgb_size; i++)
        h_rgb[i] = rand() % 256;

    color2gray_cpu(h_rgb, h_gray_cpu, width, height);
    color2gray_gpu(h_rgb, h_gray_gpu, width, height);

    int mismatch = 0;
    for (int i = 0; i < width * height; i++) {
        if (abs((int)h_gray_cpu[i] - (int)h_gray_gpu[i]) > 1) {
            mismatch++;
        }
    }

    if (mismatch == 0)
        printf("[PASS] GPU result matches CPU.\n");
    else
        printf("[FAIL] %d mismatches found.\n", mismatch);

    free(h_rgb);
    free(h_gray_cpu);
    free(h_gray_gpu);
    return 0;
}
