# 💻 Trabalho Final de Programação Paralela e Distribuída (PPD)

## 📋 Descrição
Este projeto tem como objetivo estudar e comparar diferentes abordagens para a multiplicação de inteiros de grande porte usando o **Algoritmo de Karatsuba**, explorando desde a versão sequencial até paralelizações em memória compartilhada (threads) e aceleradores (CUDA).

## 👥 Equipe
| Nome                                      | RA     |
|-------------------------------------------|--------|
| Enzo Youji Murayama                       | 813606 |
| Daniel de Souza Sobrinho Macedo           | 813524 |
| Gabriel Henrique Alves Zago               | 811640 |
| Laysson Santos da Silva                   | 800349 |

## 🏗 Estrutura do Projeto
O projeto está organizado em um notebook Jupyter/Colab com as seguintes seções principais:

1. **Algoritmo Original**  
   - Apresentação do método de Karatsuba clássico  
   - Exemplo numérico passo a passo  

2. **Implementação Paralela com Threads (OpenMP)**
   - Decomposição de dados (SPMD)  
   - Região paralela no `main` (`#pragma omp parallel` / `single`)  
   - Medição de speedup em CPU multicore  

3. **Implementações com CUDA**
   - Versão v2.0 e v2.1  
   - Versão v3.0 e v3.1  
   - Análise de desempenho e speedups obtidos  

4. **Resultados e Gráficos**
   - Comparação entre sequencial, threads e CUDA  
   - Curvas de speedup e eficiência  

5. **Referências**
   - Listagem de artigos, RFCs e documentações consultadas

Nesse repositório também possui as 6 versões da Implementação em CUDA do problema

## 🛠 Tecnologias e Ferramentas
- **Linguagens:** C/C++ (OpenMP, CUDA), Python (para análise e plotagem)  
- **Ambiente de Execução:** Google Colab, com suporte a compiladores GCC/G++ e NVCC  
- **Bibliotecas Python:**  
  - `matplotlib` / `pandas` (para gráficos e tabelas)  
  - (opcional) `ipywidgets` para parâmetros interativos  

## 🚀 Como Executar
1. Abra o notebook `PPD_Final_Project.ipynb` no **Google Colab**.  
2. **Compile** as células com código C/C++/CUDA (use `%%bash` ou `%%cuda` conforme configurado).  
3. Ajuste os tamanhos dos inteiros de teste e o número de threads/blocos conforme desejado.  
4. Execute todas as células para gerar os gráficos de speedup e tabelas de desempenho.  

> **Dica:** para testar localmente, instale o CUDA Toolkit e o OpenMP no seu sistema:
> ```bash
> sudo apt-get update
> sudo apt-get install build-essential libomp-dev nvidia-cuda-toolkit
> ```
> e compile com:
> ```bash
> g++ -fopenmp karatsuba_openmp.cpp -o karatsuba_omp
> nvcc karatsuba_cuda.cu -o karatsuba_cuda
> ```


