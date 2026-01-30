#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
int N = 128; 
int itmax = 10;
double maxeps = 1e-7;
int main(int argc, char **argv) {
    int rank, size;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-it") == 0 && i+1 < argc) {
            itmax = atoi(argv[++i]);
        }
    }
    
    if (rank == 0) {
        printf("=== MPI JACOBI SCALING ===\n");
        printf("Grid: %d^3, Processes: %d, Iterations: %d\n", N, size, itmax);
        start_time = MPI_Wtime();
    }
    
    int layers = N - 2;
    int layers_per_proc = layers / size;
    int extra_layers = layers % size;
    
    int start_layer = 1;
    for (int i = 0; i < rank; i++) {
        start_layer += layers_per_proc + (i < extra_layers ? 1 : 0);
    }
    
    int end_layer = start_layer + layers_per_proc + (rank < extra_layers ? 1 : 0);
    int my_layers = end_layer - start_layer;
    
    int total_layers = my_layers + 2;
    double (*A)[N][N] = malloc(total_layers * sizeof(*A));
    double (*B)[N][N] = malloc(total_layers * sizeof(*A));
    
    if (A == NULL || B == NULL) {
        fprintf(stderr, "Process %d: Memory allocation failed!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for (int k = 0; k < total_layers; k++) {
        int global_k = start_layer - 1 + k;
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                if (i == 0 || i == N-1 || j == 0 || j == N-1 || 
                    global_k == 0 || global_k == N-1) {
                    A[k][j][i] = 0.0;
                } else {
                    A[k][j][i] = 4.0 + i + j + global_k;
                }
            }
        }
    }
    
    double eps;
    for (int it = 1; it <= itmax; it++) {
        double local_eps = 0.0;
        
        int up_neigh = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int down_neigh = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
        
        MPI_Status status;
        
        if (up_neigh != MPI_PROC_NULL) {
            MPI_Sendrecv(&A[1][0][0], N*N, MPI_DOUBLE, up_neigh, 0,
                        &A[0][0][0], N*N, MPI_DOUBLE, up_neigh, 1,
                        MPI_COMM_WORLD, &status);
        }
        
        if (down_neigh != MPI_PROC_NULL) {
            MPI_Sendrecv(&A[total_layers-2][0][0], N*N, MPI_DOUBLE, down_neigh, 1,
                        &A[total_layers-1][0][0], N*N, MPI_DOUBLE, down_neigh, 0,
                        MPI_COMM_WORLD, &status);
        }
        
        for (int k = 1; k <= total_layers - 2; k++) {
            for (int j = 1; j < N-1; j++) {
                for (int i = 1; i < N-1; i++) {
                    B[k][j][i] = (A[k-1][j][i] + A[k+1][j][i] +
                                  A[k][j-1][i] + A[k][j+1][i] +
                                  A[k][j][i-1] + A[k][j][i+1]) / 6.0;
                }
            }
        }
        
        for (int k = 1; k <= total_layers - 2; k++) {
            for (int j = 1; j < N-1; j++) {
                for (int i = 1; i < N-1; i++) {
                    double e = fabs(A[k][j][i] - B[k][j][i]);
                    A[k][j][i] = B[k][j][i];
                    if (e > local_eps) local_eps = e;
                }
            }
        }
        
        MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        if (eps < maxeps) break;
    }
    
    double local_sum = 0.0;
    for (int k = 1; k <= total_layers - 2; k++) {
        int global_k = start_layer + k - 2;
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                local_sum += A[k][j][i] * (i+1) * (j+1) * (global_k+1) / (N*N*N);
            }
        }
    }
    
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        end_time = MPI_Wtime();
        double elapsed = end_time - start_time;
        
        printf("RESULTS: S=%.6f, Time=%.3fs, N=%d, P=%d\n", 
               global_sum, elapsed, N, size);
    }
    
    free(A);
    free(B);
    MPI_Finalize();
    return 0;
}

