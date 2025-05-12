#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_DIGITS 2097152 // 2MB of digits
#define TILE_SIZE 512
#define CUTOFF 512 // Cutoff for switching to grade school multiplication

// Kernel for grade school multiplication
__global__ void multiplyKernel_optimized(int *a, int *b, int *result, int digits) {
    __shared__ int shared_a[TILE_SIZE];
    __shared__ int shared_b[TILE_SIZE];

    int a_tile_idx = blockIdx.x;
    int a_start = a_tile_idx * TILE_SIZE;
    int tid = threadIdx.x;

    if (a_start + tid < digits)
        shared_a[tid] = a[a_start + tid];
    else
        shared_a[tid] = 0;
    __syncthreads();

    int num_b_tiles = (digits + TILE_SIZE - 1) / TILE_SIZE;
    for (int b_tile_idx = 0; b_tile_idx < num_b_tiles; b_tile_idx++) {
        int b_start = b_tile_idx * TILE_SIZE;
        if (b_start + tid < digits)
            shared_b[tid] = b[b_start + tid];
        else
            shared_b[tid] = 0;
        __syncthreads();

        int sum1 = 0, sum2 = 0;
        int k1 = tid;
        if (k1 < 2 * TILE_SIZE) {
            int i_min = (k1 >= TILE_SIZE) ? k1 - (TILE_SIZE - 1) : 0;
            int i_max = (k1 < TILE_SIZE) ? k1 : TILE_SIZE - 1;
            for (int i = i_min; i <= i_max; i++) {
                int j = k1 - i;
                sum1 += shared_a[i] * shared_b[j];
            }
        }

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

        int base_idx = a_start + b_start;
        if (base_idx + tid < 2 * digits) {
            atomicAdd(&result[base_idx + tid], sum1);
        }
        if (base_idx + tid + TILE_SIZE < 2 * digits) {
            atomicAdd(&result[base_idx + tid + TILE_SIZE], sum2);
        }
        __syncthreads();
    }
}

// Kernel for Karatsuba multiplication with recursive calls
__global__ void karatsubaKernel(int *a, int *b, int *result, int digits, int cutoff) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { // Ensure only one thread executes
        if (digits <= cutoff) {
            // Base case: use grade school multiplication
            int num_blocks = (digits + TILE_SIZE - 1) / TILE_SIZE;
            multiplyKernel_optimized<<<num_blocks, TILE_SIZE>>>(a, b, result, digits);
            cudaDeviceSynchronize(); // Wait for grade school kernel to finish
        } else {
            int half = digits / 2;

            // Pointers for high and low parts
            int *a_low = a;
            int *a_high = a + half;
            int *b_low = b;
            int *b_high = b + half;

            // Allocate memory for intermediate results
            int *z0, *z1, *z2, *a_sum, *b_sum;
            cudaMalloc(&z0, 2 * half * sizeof(int));
            cudaMalloc(&z1, 2 * half * sizeof(int));
            cudaMalloc(&z2, 2 * (digits - half) * sizeof(int));
            cudaMalloc(&a_sum, half * sizeof(int));
            cudaMalloc(&b_sum, half * sizeof(int));

            // Initialize memory
            cudaMemset(z0, 0, 2 * half * sizeof(int));
            cudaMemset(z1, 0, 2 * half * sizeof(int));
            cudaMemset(z2, 0, 2 * (digits - half) * sizeof(int));

            // Compute a_sum and b_sum
            for (int i = 0; i < half; i++) {
                a_sum[i] = a_high[i] + a_low[i];
                b_sum[i] = b_high[i] + b_low[i];
            }

            // Launch child Karatsuba kernels for subproblems
            karatsubaKernel<<<1, 1>>>(a_low, b_low, z0, half, cutoff);
            karatsubaKernel<<<1, 1>>>(a_high, b_high, z2, digits - half, cutoff);
            karatsubaKernel<<<1, 1>>>(a_sum, b_sum, z1, half, cutoff);
            cudaDeviceSynchronize(); // Wait for child kernels to finish

            // Combine results
            for (int i = 0; i < 2 * half; i++) {
                result[i] = z0[i];
            }
            for (int i = 0; i < 2 * (digits - half); i++) {
                result[i + 2 * half] = z2[i];
            }
            for (int i = 0; i < 2 * half; i++) {
                result[i + half] += z1[i] - z0[i] - (i < 2 * (digits - half) ? z2[i] : 0);
            }

            // Free allocated memory
            cudaFree(z0);
            cudaFree(z1);
            cudaFree(z2);
            cudaFree(a_sum);
            cudaFree(b_sum);
        }
    }
}

// Function to normalize the result (carry propagation)
void normalizeResult(int *r, int size) {
    int carry = 0;
    for (int i = 0; i < size; ++i) {
        r[i] += carry;
        carry = r[i] / 10;
        r[i] %= 10;
    }
}

// Function to generate random numbers
void randomNum(int *a, int d) {
    for (int i = 0; i < d; ++i) {
        a[i] = rand() % 10;
    }
    if (d > 0 && a[d-1] == 0) {
        a[d-1] = (rand() % 9) + 1;
    }
}

void printNum(int *a, int d)
{
    int i;
    for (i = d - 1; i > 0; i--)
        if (a[i] != 0)
            break;
    for (; i >= 0; i--)
        printf("%d", a[i]);
    printf("\n");
}

int escrever(const char *nomeArquivo, const int *vetor, int tamanho) {
    FILE *arquivo = fopen(nomeArquivo, "w");
    if (arquivo == NULL) {
        perror("Erro ao abrir o arquivo");
        return 1;
    }
    int i;
    for (i = tamanho - 1; i > tamanho / 10; i--) if (vetor[i] != 0) break;
    for (; i >= 0; i--) fprintf(arquivo, "%d", vetor[i]);
    fprintf(arquivo, "\n");
    fclose(arquivo);
    return 0;
}

int main() {
    int *a = (int*)malloc(MAX_DIGITS * sizeof(int));
    int *b = (int*)malloc(MAX_DIGITS * sizeof(int));
    int *r = (int*)calloc(2 * MAX_DIGITS, sizeof(int));

    srand(time(NULL));
    randomNum(a, MAX_DIGITS);
    randomNum(b, MAX_DIGITS);

    printNum(a, MAX_DIGITS);
    printNum(b, MAX_DIGITS);
    clock_t start = clock();

    int *d_a, *d_b, *d_r;
    cudaMalloc(&d_a, MAX_DIGITS * sizeof(int));
    cudaMalloc(&d_b, MAX_DIGITS * sizeof(int));
    cudaMalloc(&d_r, 2 * MAX_DIGITS * sizeof(int));

    cudaMemset(d_r, 0, 2 * MAX_DIGITS * sizeof(int));
    cudaMemset(d_a, 0, MAX_DIGITS * sizeof(int));
    cudaMemset(d_b, 0, MAX_DIGITS * sizeof(int));

    cudaMemcpy(d_a, a, MAX_DIGITS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, MAX_DIGITS * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the Karatsuba kernel with 1 block and 1 thread
    karatsubaKernel<<<1, 1>>>(d_a, d_b, d_r, MAX_DIGITS, CUTOFF);
    cudaDeviceSynchronize(); // Ensure all recursive calls and children finish

    cudaMemcpy(r, d_r, 2 * MAX_DIGITS * sizeof(int), cudaMemcpyDeviceToHost);
    normalizeResult(r, 2 * MAX_DIGITS);
    escrever("cuda.txt", r, 2*MAX_DIGITS);

    printf("Time: %f ms\n", ((double)(clock() - start))*1000 / CLOCKS_PER_SEC);

    free(a);
    free(b);
    free(r);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_r);

    return 0;
}