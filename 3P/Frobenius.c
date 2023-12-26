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


    if (argc < 3) {
        printf("Es necesario definir el tamaño de la matriz y el paso de la distribución.\n");
        return EXIT_FAILURE;
    }

    long N = atoi(argv[1]); // Tamaño de la matriz
    long F = atoi(argv[2]); // Paso de la distribución cíclica

    if (N <= 0) {
        printf("El tamaño de a matriz debe de ser mayor que cero.\n");
        return EXIT_FAILURE;
    }

    gettimeofday(&t_prev,NULL);
    gettimeofday(&t_init,NULL);

    double **A = allocate_matrix(N, N);

    int node = 0, npes;
    double s_local = 0.0, s = 0.0, norm = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

	
    if (npes > N) {
        printf("Número de nodos excesivo para el tamaño de la matriz.\n");
        printf("Reduzca el número de nodos.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if(F > N/npes) {
        printf("El tamaño del paso es excesivo.\n");
        printf("El tamaño máximo permitido para este caso es: %ld\n",N/npes);
        printf("Considere reducir el paso.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    // Proceso 0 inicializa y envía la matriz A a todos los procesos
    if (!node) {
        // Inicialización de la matriz A (aquí puedes cargarla desde un archivo o generarla)
        for (long i = 0; i < N; i++) {
            for (long j = 0; j < N; j++) {
                //A[i][j] = (double)rand()/(double) RAND_MAX; // Valores aleatorios entre 0 y 1
		A[i][j] = 1.0;
            }
        }
    }

    // Envío el tamaño de la Matriz N a todos los procesos
    MPI_Bcast(&N, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // Envío el paso F de la distribución cíclica a todos los procesos
    MPI_Bcast(&F, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    short start = 0;

    if(!node) {
	    
        // Empieza a trabajar cuando el nodo 1 acaba la función allocate_matrix()
        MPI_Recv(&start, 1, MPI_SHORT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
        
	long fila = 0;
	    
        for (int nodo_dest = 1; nodo_dest < npes; nodo_dest++){
            for (long i = 0; i < F; i++){
                fila = (nodo_dest - 1) * F + i;
                printf("%d: row = %ld\n", node, fila);
                MPI_Send(A[fila], N, MPI_DOUBLE, nodo_dest, nodo_dest, MPI_COMM_WORLD);
            }

        }
    } else {
	    
        double **A = allocate_matrix(F, N);
	
        if(node == 1) {
            MPI_Send(&start, 1, MPI_SHORT, 0, 1, MPI_COMM_WORLD);
        }

        for (long i = 0; i < F; i++) {
            MPI_Recv(A[i], N, MPI_DOUBLE, 0, node, MPI_COMM_WORLD, NULL);
        }

    }


    // Cada proceso calcula localmente la suma de los cuadrados de sus elementos asignados

    if (!node) {
        // El número de procesos tiene un número de filas asignado que es menor
        // que el tamaño en filas de la matriz. El nodo 0 se hará cargo de las filas
        // restantes
        if(F*(npes-1) < N){
            for(long i = F*(npes-1); i < N; i ++){
                for (long j = 0; j < N; j++) {
                    s_local += A[i][j] * A[i][j];
                }
            }
        }
	    printf("S_local node 0: %f\n",s_local);
    }else{
        for (long i = 0; i < F; i++) {
            for (long j = 0; j < N; j++) {
		printf("%f",A[i][j]);
                s_local += A[i][j] * A[i][j];
            }
        }
	    printf("\nS_local: %f\n",s_local);
    }
    
    // Reducción para obtener la suma total s en el proceso 0
    MPI_Reduce(&s_local, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    gettimeofday(&t_final, NULL);
    overhead = (t_init.tv_sec-t_prev.tv_sec+(t_init.tv_usec-t_prev.tv_usec)/1.e6);
    total_time = (t_final.tv_sec-t_init.tv_sec+(t_final.tv_usec-t_init.tv_usec)/1.e6)-overhead;

    // Proceso 0 recibe las sumas locales y calcula la norma de Frobenius
    if (node == 0) {
        
        norm = sqrt(s);

	printf("---------------------- RESULTS -----------------------\n");
        printf("Frobenius Norm == %.50f\n", norm);
        printf("Time == %.50f\n", total_time);
	    printf(" Rows per proc = %ld\n", F);
        printf("------------------------------------------------------\n");

        // Create CSV file
        FILE* fp = fopen("results_frobenius.csv", "a");

        if (fp == NULL) {
            printf("Error opening file results.csv\n");
            exit(EXIT_FAILURE);
        }
        setlocale(LC_ALL, "es_ES.utf8");

        fprintf(fp,"%d;%.50f;%.50f;%ld\n", npes, total_time, norm, F);

        // Closing CSV file
        fclose(fp); 
        free_matrix(A, N);
    }else{
        free_matrix(A, F);
    }

    MPI_Finalize();
    return 0;
}
