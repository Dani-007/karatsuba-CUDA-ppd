#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define MAX_DIGITS 2097152 // 2MB de dígitos
#define KARAT_CUTOFF 2048
#define POOL_SIZE (4ULL * (size_t)MAX_DIGITS * ((size_t)log2(MAX_DIGITS) + 1024ULL)) // Tamanho do pool de memória

// Estrutura para gerenciar o pool de memória
typedef struct
{
  int *data;     // Memória global na GPU
  size_t offset; // Offset atual
  size_t *stack; // Stack para gerenciar offsets
  int stack_top; // Topo da stack
} MemPool;

__global__ void gradeSchool(int *a, int *b, int *ret, int d);
__global__ void computeAsumBsum(int *a, int *al, int *ar, int *b, int *bl, int *br, int *asum, int *bsum, int d);
__global__ void combineResults(int *x1, int *x2, int *x3, int *ret, int d);
__global__ void addVectors(int *dest, int *src, int offset, int n);

void randomNum(int *a, int *d);
void karatsuba(MemPool *pool, int *a, int *b, int *ret, int d);
void doCarry(int *a, int d);
void printNum(int *a, int d);

int *a;
int *b;
int *r;

void pool_init(MemPool *pool, size_t total_size)// Inicializar pool de memória
{
  cudaMalloc(&pool->data, total_size * sizeof(int));
  cudaMallocHost(&pool->stack, 1000 * sizeof(size_t));
  pool->stack_top = -1;
  pool->offset = 0;
}

void pool_free(MemPool *pool)// Liberar pool de memória baseado na stack
{
  pool->offset = pool->stack[pool->stack_top--];
}

// Função para alocar do pool baseado no offset
int *pool_alloc(MemPool *pool, size_t n)
{
  pool->stack[++pool->stack_top] = pool->offset;

  // Usar size_t para comparação
  if (pool->offset + n > (size_t)POOL_SIZE)
  {
    fprintf(stderr, "Pool overflow: %zu > %zu\n",
            pool->offset + n, (size_t)POOL_SIZE);
    exit(1);
  }

  int *ptr = pool->data + pool->offset;
  pool->offset += n;
  return ptr;
}

int main()
{
  int d_a = MAX_DIGITS, d_b = MAX_DIGITS, d, i;

  a = (int *)malloc(MAX_DIGITS * sizeof(int));
  b = (int *)malloc(MAX_DIGITS * sizeof(int));
  r = (int *)malloc(6 * MAX_DIGITS * sizeof(int));

  if (!a || !b || !r)// Verificar alocação
  {
    fprintf(stderr, "Falha na alocação host\n");
    exit(1);
  }

  // Inicialização dos números
  srand(time(NULL));
  randomNum(a, &d_a);
  randomNum(b, &d_b);

  // Ajustar para potência de 2
  for (d = 1; d < (d_a > d_b ? d_a : d_b); d *= 2)
    ;
  for (i = d_a; i < d; a[i++] = 0)
    ;
  for (i = d_b; i < d; b[i++] = 0)
    ;

  clock_t start = clock();
  // Alocar pool de memória na GPU
  int recursion_depth = (int)(log2(d) - log2(KARAT_CUTOFF)) + 1;

  MemPool pool;// Inicializar pool de memória
  size_t log_term = (size_t)(log2(MAX_DIGITS) + 1);// Tamanho do pool de memória
  size_t pool_size = 12 * MAX_DIGITS * (size_t)(log2(MAX_DIGITS) + 1);// Tamanho do pool de memória
  printf("Alocando %zu ints na GPU...\n", pool_size);
  pool_init(&pool, pool_size);// Inicializar pool de memória
  printf("Pool alocado com sucesso!\n");

  int *dev_a, *dev_b, *dev_r;
  cudaMalloc(&dev_a, MAX_DIGITS * sizeof(int));
  cudaMalloc(&dev_b, MAX_DIGITS * sizeof(int));
  cudaMalloc(&dev_r, 6 * MAX_DIGITS * sizeof(int));

  cudaMemcpy(dev_a, a, MAX_DIGITS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, MAX_DIGITS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(dev_r, 0, 6 * MAX_DIGITS * sizeof(int));

  karatsuba(&pool, dev_a, dev_b, dev_r, d);

  cudaMemcpy(r, dev_r, 6 * MAX_DIGITS * sizeof(int), cudaMemcpyDeviceToHost);
  doCarry(r, 2 * d);
  
  // printf("Number 1: ");
  // printNum(a, d_a);
  // printf("Number 2: ");
  // printNum(b, d_b);
  // printf("Result: ");
  // printNum(r, 2 * MAX_DIGITS);

  printf("Time: %f\n", ((double)(clock() - start)) / CLOCKS_PER_SEC);

  cudaFree(pool.data);
  cudaFreeHost(pool.stack);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_r);

  free(a);
  free(b);
  free(r);
  return 0;
}

void doCarry(int *a, int d)
{
  int c;
  int i;

  c = 0;
  for (i = 0; i < d; i++)
  {
    a[i] += c;
    if (a[i] < 0)
    {
      c = -(-(a[i] + 1) / 10 + 1);
    }
    else
    {
      c = a[i] / 10;
    }
    a[i] -= c * 10;
  }
  if (c != 0)
    fprintf(stderr, "Overflow %d\n", c);
}

void randomNum(int *a, int *d)
{
  for (int i = 0; i < *d; i++)
  {
    a[i] = rand() % 10;
  }

  if (*d > 0 && a[*d - 1] == 0)
  {
    a[*d - 1] = (rand() % 9) + 1;
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

void getNum(int *a, int *d_a)
{
  int c;
  int i;

  *d_a = 0;
  while (true)
  {
    c = getchar();
    if (c == '\n' || c == EOF)
      break;
    if (*d_a >= MAX_DIGITS)
    {
      fprintf(stderr, "using only first %d digits\n", MAX_DIGITS);
      while (c != '\n' && c != EOF)
        c = getchar();
    }
    a[*d_a] = c - '0';
    ++(*d_a);
  }
  // reverse the number so that the 1's digit is first
  for (i = 0; i * 2 < *d_a - 1; i++)
  {
    c = a[i], a[i] = a[*d_a - i - 1], a[*d_a - i - 1] = c;
  }
}

// ret must have space for 6d digits.
// the result will be in only the first 2d digits
// my use of the space in ret is pretty creative.
// | ar*br  | al*bl  | asum*bsum | lower-recursion space | asum | bsum |
//  d digits d digits  d digits     3d digits              d/2    d/2
// Modifique a função karatsuba para executar na GPU
void karatsuba(MemPool *pool, int *a, int *b, int *ret, int d)
{
  if (d <= KARAT_CUTOFF)
  {
    int threads = 256;
    int max_blocks = 65535;
    int blocks = min((d * d + threads - 1) / threads, max_blocks);
    gradeSchool<<<blocks, threads>>>(a, b, ret, d);
    cudaDeviceSynchronize();
    return;
  }

  // Aloca na memória da GPU 
  int *x1 = pool_alloc(pool, d);
  if (!x1)
  {
    printf("Falha ao alocar x1\n");
    exit(1);
  }
  int *x2 = pool_alloc(pool, d);
  if (!x2)
  {
    printf("Falha ao alocar x2\n");
    exit(1);
  }
  int *x3 = pool_alloc(pool, d);
  if (!x3)
  {
    printf("Falha ao alocar x3\n");
    exit(1);
  }
  int *asum = pool_alloc(pool, d / 2);
  if (!asum)
  {
    printf("Falha ao alocar asum\n");
    exit(1);
  }
  int *bsum = pool_alloc(pool, d / 2);
  if (!bsum)
  {
    printf("Falha ao alocar bsum\n");
    exit(1);
  }

  // Inicializar memória
  cudaMemset(x1, 0, d * sizeof(int));// a_H * b_H
  cudaMemset(x2, 0, d * sizeof(int));// a_L * b_L
  cudaMemset(x3, 0, d * sizeof(int));// (a_H + a_L) * (b_H + b_L)

  // Configurar ponteiros
  int *ar = a, *al = a + d / 2;
  int *br = b, *bl = b + d / 2;

  // Calcular asum/bsum
  dim3 block(256), grid((d / 2 + 255) / 256);
  computeAsumBsum<<<grid, block>>>(a, al, ar, b, bl, br, asum, bsum, d);//
  cudaDeviceSynchronize();

  // Chamadas recursivas
  karatsuba(pool, ar, br, x1, d / 2);
  karatsuba(pool, al, bl, x2, d / 2);
  karatsuba(pool, asum, bsum, x3, d / 2);

  // Combinar resultados
  addVectors<<<grid, block>>>(ret, x1, 0, d);
  addVectors<<<grid, block>>>(ret, x2, d, d);
  combineResults<<<grid, block>>>(x1, x2, x3, ret, d);
  cudaDeviceSynchronize();

  pool_free(pool);
}

__global__ void gradeSchool(int *a, int *b, int *ret, int d)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < d * d)
  {
    int i = idx / d;
    int j = idx % d;
    atomicAdd(&ret[i + j], a[i] * b[j]);
  }
}

__global__ void computeAsumBsum(int *a, int *al, int *ar, int *b, int *bl, int *br, int *asum, int *bsum, int d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d / 2)
  {
    asum[i] = al[i] + ar[i];
    bsum[i] = bl[i] + br[i];
  }
}

__global__ void combineResults(int *x1, int *x2, int *x3, int *ret, int d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d)
  {
    x3[i] = x3[i] - x1[i] - x2[i];
    atomicAdd(&ret[i + d / 2], x3[i]);
  }
}

__global__ void addVectors(int *dest, int *src, int offset, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    atomicAdd(&dest[offset + i], src[i]);
  }
}