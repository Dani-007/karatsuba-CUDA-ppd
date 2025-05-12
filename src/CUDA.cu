#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_DIGITS 2097152 // 2MB de dígitos

__global__ void gradeSchoolKernel(int *a, int *b, int *ret, int d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d)
  {
    for (int j = 0; j < d; j++)
    {
      atomicAdd(&ret[i + j], a[i] * b[j]); // Multiplicação básica
    }
  }
}

void normalizeResult(int *r, int size)
{
  for (int i = 0; i < size - 1; i++)
  {
    r[i + 1] += r[i] / 10;
    r[i] %= 10;
  }
}

void gradeSchoolCUDA(int *a, int *b, int *ret, int d)
{
  int *d_a, *d_b, *d_ret;
  size_t size = d * sizeof(int);
  size_t ret_size = (2 * d) * sizeof(int);

  if (cudaMalloc(&d_a, size) != cudaSuccess ||
      cudaMalloc(&d_b, size) != cudaSuccess ||
      cudaMalloc(&d_ret, ret_size) != cudaSuccess)
  {
    printf("Erro ao alocar memória na GPU.\n");
    return;
  }

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  cudaMemset(d_ret, 0, ret_size);

  int threadsPerBlock = 1024;
  int blocks = (d + threadsPerBlock - 1) / threadsPerBlock;

  gradeSchoolKernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_ret, d);
  cudaDeviceSynchronize();

  cudaMemcpy(ret, d_ret, ret_size, cudaMemcpyDeviceToHost);
  normalizeResult(ret, 2 * d);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_ret);
}

void randomNum(int *a, int d)
{
  for (int i = 0; i < d; i++)
  {
    a[i] = rand() % 10;
  }
  if (d > 0 && a[d - 1] == 0)
  {
    a[d - 1] = (rand() % 9) + 1;
  }
}

void printNum(int *a, int d)
{
  int start = d - 1;
  while (start > 0 && a[start] == 0)
    start--;
  for (int i = start; i >= 0; i--)
  {
    printf("%d", a[i]);
  }
  printf("\n");
}

int main()
{
  int *a = (int *)malloc(MAX_DIGITS * sizeof(int));
  int *b = (int *)malloc(MAX_DIGITS * sizeof(int));
  int *r = (int *)calloc(2 * MAX_DIGITS, sizeof(int));

  if (!a || !b || !r)
  {
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
  printf("Running grade school multiplication...\n");
  start = clock();
  gradeSchoolCUDA(a, b, r, MAX_DIGITS);
  end = clock();

  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time: %f segundos\n", cpu_time_used);

  // Verificação
  // printf("Verifying result...\n");
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
