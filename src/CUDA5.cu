#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// nvcc -arch=sm_86 -O3 -o gradeSchoolDan gradeSchoolDan.cu

#define MAX_DIGITS 2097152 // 2MB de dígitos
#define TILE_SIZE 512

// Kernel otimizado que utiliza melhor a memória compartilhada e reduz chamadas de atomicAdd
__global__ void multiplyKernel_optimized(int *a, int *b, int *result, int digits) {
    __shared__ int shared_a[TILE_SIZE];
    __shared__ int shared_b[TILE_SIZE];

    int a_tile_idx = blockIdx.x;
    int a_start = a_tile_idx * TILE_SIZE;
    int tid = threadIdx.x;

    // Carrega o tile de A para o bloco
    if (a_start + tid < digits)
        shared_a[tid] = a[a_start + tid];
    else
        shared_a[tid] = 0;
    __syncthreads();

    int num_b_tiles = (digits + TILE_SIZE - 1) / TILE_SIZE;
    for (int b_tile_idx = 0; b_tile_idx < num_b_tiles; b_tile_idx++) {
        int b_start = b_tile_idx * TILE_SIZE;
        // Carrega o tile de B para o bloco
        if (b_start + tid < digits)
            shared_b[tid] = b[b_start + tid];
        else
            shared_b[tid] = 0;
        __syncthreads();

        // Cada iteração calcula um tile parcial do produto (tamanho 2*TILE_SIZE).
        // Cada thread calculará até duas posições no tile, evitando chamadas repetidas a atomicAdd.
        int sum1 = 0, sum2 = 0;

        // Primeira posição: k = tid
        int k1 = tid;
        if (k1 < 2 * TILE_SIZE) {
            int i_min = (k1 >= TILE_SIZE) ? k1 - (TILE_SIZE - 1) : 0;
            int i_max = (k1 < TILE_SIZE) ? k1 : TILE_SIZE - 1;
            for (int i = i_min; i <= i_max; i++) {
                int j = k1 - i;
                sum1 += shared_a[i] * shared_b[j];
            }
        }

        // Segunda posição: k = tid + TILE_SIZE
        int k2 = tid + TILE_SIZE;
        if (k2 < 2 * TILE_SIZE) {
            int i_min = (k2 >= TILE_SIZE) ? k2 - (TILE_SIZE - 1) : 0;
            int i_max = (k2 < TILE_SIZE) ? k2 : TILE_SIZE - 1;
            for (int i = i_min; i <= i_max; i++) {
                int j = k2 - i;
                sum2 += shared_a[i] * shared_b[j];
            }
        }
        __syncthreads();

        // Índice base global para este tile parcial
        int base_idx = a_start + b_start;
        // Cada thread atualiza até duas posições do resultado global com uma única chamada atomicAdd por posição.
        if (base_idx + tid < 2 * digits) {
            atomicAdd(&result[base_idx + tid], sum1);
        }
        if (base_idx + tid + TILE_SIZE < 2 * digits) {
            atomicAdd(&result[base_idx + tid + TILE_SIZE], sum2);
        }
        __syncthreads();
    }
}

void normalizeResult(int *r, int size) {
    int carry = 0;
    for (int i = 0; i < size; ++i) {
        r[i] += carry;
        carry = r[i] / 10;
        r[i] %= 10;
    }
}

void randomNum(int *a, int d) {
    for (int i = 0; i < d; ++i) {
        a[i] = rand() % 10;
    }
    if (d > 0 && a[d-1] == 0) {
        a[d-1] = (rand() % 9) + 1;
    }
}

int main() {
    int *a = (int*)malloc(MAX_DIGITS * sizeof(int));
    int *b = (int*)malloc(MAX_DIGITS * sizeof(int));
    int *r = (int*)calloc(2 * MAX_DIGITS, sizeof(int));

    srand(time(NULL));
    randomNum(a, MAX_DIGITS);
    randomNum(b, MAX_DIGITS);
    clock_t start = clock();

    int *d_a, *d_b, *d_r;
    cudaMalloc(&d_a, MAX_DIGITS * sizeof(int));
    cudaMalloc(&d_b, MAX_DIGITS * sizeof(int));
    cudaMalloc(&d_r, 2 * MAX_DIGITS * sizeof(int));

    cudaMemcpy(d_a, a, MAX_DIGITS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, MAX_DIGITS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_r, 0, 2 * MAX_DIGITS * sizeof(int));

    int num_blocks = (MAX_DIGITS + TILE_SIZE - 1) / TILE_SIZE;
    multiplyKernel_optimized<<<num_blocks, TILE_SIZE>>>(d_a, d_b, d_r, MAX_DIGITS);

    cudaMemcpy(r, d_r, 2 * MAX_DIGITS * sizeof(int), cudaMemcpyDeviceToHost);
    normalizeResult(r, 2 * MAX_DIGITS);

    // Trechos de verificação (descomentáveis se necessário)
    // printf("Número 1: ");
    // for (int i = MAX_DIGITS-1; i >= 0; i--) printf("%d", a[i]);
    // printf("\nNúmero 2: ");
    // for (int i = MAX_DIGITS-1; i >= 0; i--) printf("%d", b[i]);
    // printf("\nResultado: ");
    // for (int i = 2*MAX_DIGITS-1; i >= 0; i--) printf("%d", r[i]);

    printf("\n");
    printf("Time: %f\n", ((double)(clock() - start))*1000 / CLOCKS_PER_SEC);

    free(a); free(b); free(r);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
    return 0;
}
