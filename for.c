#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))

double maxeps = 0.1e-7;
int itmax = 10;

void relax(double *src, double *dst, int N, double *eps);
void init(double *A, int N);
void verify(double *A, int N, double *s);
double* allocate_matrix(int N);
void free_matrix(double *A);

int main(int an, char **as)
{
    int num_threads = 1;
    if (an > 1) {
        num_threads = atoi(as[1]);
    }
    omp_set_num_threads(num_threads);

    int base_N = 32;
    int sizes[] = {1, 2, 3, 4, 6, 8};
    int num_sizes = 6;

    printf("Threads,Size,N,Time,Sum,Version\n");

    for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
        int N = base_N * sizes[size_idx];
        double *A = allocate_matrix(N);
        double *B = allocate_matrix(N);

        double start_time = omp_get_wtime();
        
        init(A, N);
        
        for(int it = 1; it <= itmax; it++) {
            double eps = 0.;
            if (it % 2 == 1) {
                relax(A, B, N, &eps);
            } else {
                relax(B, A, N, &eps);
            }
            if (eps < maxeps) break;
        }
        
        double s;
        verify(A, N, &s);
        double end_time = omp_get_wtime();
        
        printf("%d,%d,%d,%.6f,%.6f,for\n", 
               num_threads, sizes[size_idx], N, end_time - start_time, s);

        free_matrix(A);
        free_matrix(B);
    }
    
    return 0;
}

double* allocate_matrix(int N) {
    return (double*)calloc(N * N * N, sizeof(double));
}

void free_matrix(double *A) {
    free(A);
}

void init(double *A, int N) {
    #pragma omp parallel for collapse(3) schedule(static)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < N; k++) {
                int idx = i * N * N + j * N + k;
                if(i == 0 || i == N-1 || j == 0 || j == N-1 || k == 0 || k == N-1) 
                    A[idx] = 0.;
                else 
                    A[idx] = (4. + i + j + k);
            }
        }
    }
}

void relax(double *src, double *dst, int N, double *eps) {
    int num_threads = omp_get_max_threads();
    double *local_eps_array = (double*)calloc(num_threads, sizeof(double));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for collapse(3) schedule(static) nowait
        for(int i = 1; i <= N-2; i++) {
            for(int j = 1; j <= N-2; j++) {
                for(int k = 1; k <= N-2; k++) {
                    int idx = i * N * N + j * N + k;
                    
                    double new_val = (src[(i-1) * N * N + j * N + k] + src[(i+1) * N * N + j * N + k] + 
                                    src[i * N * N + (j-1) * N + k] + src[i * N * N + (j+1) * N + k] + 
                                    src[i * N * N + j * N + (k-1)] + src[i * N * N + j * N + (k+1)]) / 6.;
                    
                    dst[idx] = new_val;
                    
                    double diff = fabs(src[idx] - new_val);
                    if(diff > local_eps_array[tid]) local_eps_array[tid] = diff;
                }
            }
        }
    }
    
    *eps = 0.0;
    for(int t = 0; t < num_threads; t++) {
        if(local_eps_array[t] > *eps) {
            *eps = local_eps_array[t];
        }
    }
    free(local_eps_array);
}

void verify(double *A, int N, double *s) {
    double local_s = 0.;
    
    #pragma omp parallel for collapse(3) schedule(static) reduction(+:local_s)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < N; k++) {
                int idx = i * N * N + j * N + k;
                local_s += A[idx] * (i+1) * (j+1) * (k+1) / (N*N*N);
            }
        }
    }
    
    *s = local_s;
}

