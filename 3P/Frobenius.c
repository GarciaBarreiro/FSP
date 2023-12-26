#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <locale.h>


double **allocate_matrix(int rows, int cols) {
    double **matrix = (double **)malloc(rows * sizeof(double *));
    if (matrix == NULL) {
        fprintf(stderr, "Error de asignación de memoria.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double *)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Error de asignación de memoria.\n");
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

void free_matrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main(int argc, char *argv[]) {

    struct timeval t_prev, t_init, t_final;
    double overhead, total_time;


if (argc != 3) {
        printf("Uso: mpirun -np <npes> programa <N> <F>\n");
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]); // Tamaño de la matriz
    int F = atoi(argv[2]); // Paso de la distribución cíclica

    if (N <= 0) {
        printf("El tamaño de a matriz debe de ser mayor que cero.\n");
        return EXIT_FAILURE;
    }

    if (F > N/2) {
        printf("El tamaño del paso es excesivo.\n");
        return EXIT_FAILURE;
    }


    gettimeofday(&t_prev,NULL);
    gettimeofday(&t_init,NULL);

    double **A = allocate_matrix(N, N);

    int i, j, node, npes;
    double s_local = 0.0, s = 0.0, norm = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    // Proceso 0 inicializa y envía la matriz A a todos los procesos
    if (node == 0) {
        // Inicialización de la matriz A (aquí puedes cargarla desde un archivo o generarla)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[i][j] = (double)rand()/(double) RAND_MAX; // Valores aleatorios entre 0 y 1
            }
        }
    }

    // Envío de la matriz A a todos los procesos
    MPI_Bcast(&A[0][0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Cada proceso calcula localmente la suma de los cuadrados de sus elementos asignados
    int rows_per_proc = N / npes;
    int start_row = node * rows_per_proc;
    int end_row = (node + 1) * rows_per_proc;
    if (node == npes - 1) {
        end_row = N; // El último proceso maneja las filas restantes si N no es divisible por npes
    }

    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < N; j++) {
            s_local += A[i][j] * A[i][j];
        }
    }

    // Reducción para obtener la suma total s en el proceso 0
    //MPI_Reduce(&s_local, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    overhead = (t_init.tv_sec-t_prev.tv_sec+(t_init.tv_usec-t_prev.tv_usec)/1.e6);
    total_time = (t_final.tv_sec-t_init.tv_sec+(t_final.tv_usec-t_init.tv_usec)/1.e6)-overhead;

    // Proceso 0 recibe las sumas locales y calcula la norma de Frobenius
    if (node == 0) {
        double *recv_buffer = (double *)malloc(npes * sizeof(double));
        MPI_Gather(&s, 1, MPI_DOUBLE, recv_buffer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        s = s_local;
        for (i = 0; i < npes; i++) {
            s += recv_buffer[i];
        }

        free(recv_buffer);

        norm = sqrt(s);

	printf("---------------------- RESULTS -----------------------\n");
        printf("Frobenius Norm == %.50f\n", norm);
        printf("Time == %.50f\n", total_time);
	printf(" Rows per proc = %d\n", rows_per_proc);
        printf("------------------------------------------------------\n");

        // Create CSV file
        FILE* fp = fopen("results_frobenius.csv", "a");

        if (fp == NULL) {
            printf("Error opening file results.csv\n");
            exit(EXIT_FAILURE);
        }
        setlocale(LC_ALL, "es_ES.utf8");

        fprintf(fp,"%d;%.50f;%d\n", npes, total_time, rows_per_proc);

        // Closing CSV file
        fclose(fp);

    }

    free_matrix(A, N);
    MPI_Finalize();
    return 0;
}