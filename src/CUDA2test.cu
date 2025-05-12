#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_DIGITS 2097152  // 2MB de dígitos

// Kernel para multiplicação básica (usado quando n <= THRESHOLD)
__global__ void basicMultiplyKernel(int *a, int *b, int *ret, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        for (int j = 0; j < d; j++) {
            atomicAdd(&ret[i + j], a[i] * b[j]);
        }
    }
}

void normalizeResult(int *r, int size) {
    for (int i = 0; i < size - 1; i++) {
        r[i + 1] += r[i] / 10;
        r[i] %= 10;
    }
}

void karatsubaMultiply(int *a, int *b, int *ret, int d, int threshold) {
    if (d <= threshold) {
        int *d_a, *d_b, *d_ret;
        size_t size = d * sizeof(int);
        size_t ret_size = (2 * d) * sizeof(int);

        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_ret, ret_size);

        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
        cudaMemset(d_ret, 0, ret_size);

        int threadsPerBlock = 256;
        int blocks = (d + threadsPerBlock - 1) / threadsPerBlock;
        basicMultiplyKernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_ret, d);
        cudaDeviceSynchronize();

        cudaMemcpy(ret, d_ret, ret_size, cudaMemcpyDeviceToHost);
        normalizeResult(ret, 2 * d);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_ret);
        return;
    }

    int m = d / 2;
    int *X_L = a + m;
    int *X_R = a;
    int *Y_L = b + m;
    int *Y_R = b;

    int *P1 = (int *)calloc(2 * m, sizeof(int));
    int *P2 = (int *)calloc(2 * m, sizeof(int));
    int *P3 = (int *)calloc(2 * m, sizeof(int));
    int *X_sum = (int *)calloc(m, sizeof(int));
    int *Y_sum = (int *)calloc(m, sizeof(int));

    for (int i = 0; i < m; i++) {
        X_sum[i] = X_L[i] + X_R[i];
        Y_sum[i] = Y_L[i] + Y_R[i];
    }

    karatsubaMultiply(X_L, Y_L, P1, m, threshold);
    karatsubaMultiply(X_R, Y_R, P2, m, threshold);
    karatsubaMultiply(X_sum, Y_sum, P3, m, threshold);

    for (int i = 0; i < 2 * m; i++) {
        ret[i] += P2[i];
        ret[i + m] += P3[i] - P1[i] - P2[i];
        ret[i + 2 * m] += P1[i];
    }
    normalizeResult(ret, 2 * d);

    free(P1);
    free(P2);
    free(P3);
    free(X_sum);
    free(Y_sum);
}

void randomNum(int *a, int d) {
    for (int i = 0; i < d; i++) {
        a[i] = rand() % 10;
    }
    if (d > 0 && a[d - 1] == 0) {
        a[d - 1] = (rand() % 9) + 1;
    }
}

int main() {
    FILE *file = fopen("results.csv", "w");
    if (!file) {
        printf("Erro ao criar arquivo de resultados.\n");
        return 1;
    }
    fprintf(file, "THRESHOLD,TIME\n");

    int *a = (int *)malloc(MAX_DIGITS * sizeof(int));
    int *b = (int *)malloc(MAX_DIGITS * sizeof(int));
    int *r = (int *)calloc(2 * MAX_DIGITS, sizeof(int));

    if (!a || !b || !r) {
        printf("Erro ao alocar memória.\n");
        return 1;
    }

    srand(time(NULL));
    randomNum(a, MAX_DIGITS);
    randomNum(b, MAX_DIGITS);

    int threshold = 1024;
    while (1) {
        printf("Testando THRESHOLD = %d\n", threshold);
        clock_t start = clock();
        karatsubaMultiply(a, b, r, MAX_DIGITS, threshold);
        clock_t end = clock();
        double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

        fprintf(file, "%d,%.6f\n", threshold, elapsed_time);
        fflush(file);
        printf("THRESHOLD = %d, Time = %.6f segundos\n", threshold, elapsed_time);

        if (elapsed_time >= 71.0) break;
        threshold += 256;
    }

    fclose(file);
    free(a);
    free(b);
    free(r);

    return 0;
}
