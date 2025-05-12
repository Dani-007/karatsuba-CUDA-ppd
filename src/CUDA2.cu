#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_DIGITS 2097152  // 2MB de dígitos
#define THRESHOLD 15104       // Limite para multiplicação direta na GPU

// Kernel para multiplicação básica (usado quando n <= THRESHOLD)
__global__ void basicMultiplyKernel(int *a, int *b, int *ret, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) {
        for (int j = 0; j < d; j++) {
            atomicAdd(&ret[i + j], a[i] * b[j]);  // Multiplicação básica
        }
    }
}

// Normaliza o resultado (propaga os "vai 1")
void normalizeResult(int *r, int size) {
    for (int i = 0; i < size - 1; i++) {
        r[i + 1] += r[i] / 10;
        r[i] %= 10;
    }
}

// Multiplicação Karatsuba (executada na CPU)
void karatsubaMultiply(int *a, int *b, int *ret, int d) {
    if (d <= THRESHOLD) {  // Se for pequeno, usa multiplicação normal
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

    karatsubaMultiply(X_L, Y_L, P1, m);
    karatsubaMultiply(X_R, Y_R, P2, m);
    karatsubaMultiply(X_sum, Y_sum, P3, m);

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

// Gera números aleatórios de até d dígitos
void randomNum(int *a, int d) {
    for (int i = 0; i < d; i++) {
        a[i] = rand() % 10;
    }
    if (d > 0 && a[d - 1] == 0) {
        a[d - 1] = (rand() % 9) + 1;
    }
}

// Imprime um número grande
void printNum(int *a, int d) {
    int start = d - 1;
    while (start > 0 && a[start] == 0) start--;
    for (int i = start; i >= 0; i--) {
        printf("%d", a[i]);
    }
    printf("\n");
}

// Programa principal
int main() {
    int *a = (int *)malloc(MAX_DIGITS * sizeof(int));
    int *b = (int *)malloc(MAX_DIGITS * sizeof(int));
    int *r = (int *)calloc(2 * MAX_DIGITS, sizeof(int));

    if (!a || !b || !r) {
        printf("Erro ao alocar memória.\n");
        return 1;
    }

    printf("Initializing random numbers...\n");
    srand(time(NULL));
    randomNum(a, MAX_DIGITS);
    randomNum(b, MAX_DIGITS);

    // Medição de tempo
    clock_t start, end;
    double cpu_time_used;
    printf("Running Karatsuba multiplication with CUDA...\n");
    start = clock();
    karatsubaMultiply(a, b, r, MAX_DIGITS);
    end = clock();

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %f segundos\n", cpu_time_used);

    // Verificação
    // printf("Number 1: ");
    // printNum(a, MAX_DIGITS);
    // printf("Number 2: ");
    // printNum(b, MAX_DIGITS);
    // printf("Result: ");
    // printNum(r, 2 * MAX_DIGITS);

    free(a);
    free(b);
    free(r);

    return 0;
}
