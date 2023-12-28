#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>

double **_alloc_matrix(long mat_m, long mat_n) {
    double **matrix;

    if (!(matrix = malloc(sizeof(double*)*mat_m))) {
        printf("Error allocating matrix\n");
        exit(1);
    }
    for (long i = 0; i < mat_m; i++) {
        if (!(matrix[i] = malloc(sizeof(double)*mat_n))) {
            printf("Error allocating matrix\n");
            exit(1);
        }
    }

    return matrix;
}

double **_gen_matrix(long mat_m, long mat_n) {
    double **matrix = _alloc_matrix(mat_m, mat_n);

    for (long i = 0; i < mat_m; i++) {
        for (long j = 0; j < mat_n; j++) {
            matrix[i][j] = rand() % 1000 + ((rand() % 100) / 100.0);
        }
    }
    return matrix;
}

double **_transpose(double **mat, long *mat_m, long *mat_n) {
    double **trans;
    
    if (!(trans = malloc(sizeof(double*)*(*mat_n)))) {
        printf("Error allocating transpose\n");
        exit(1);
    }

    for (long i = 0; i < *mat_n; i++) {
        if (!(trans[i] = malloc(sizeof(double)*(*mat_m)))) {
            printf("Error allocating transpose\n");
            exit(1);
        }
    }

    for (long i = 0; i < *mat_n; i++) {
        for (long j = 0; j < *mat_m; j++) {
            trans[i][j] = mat[j][i];
        }
    }
    
    int temp = *mat_m;
    *mat_m = *mat_n;
    *mat_n = temp;

    // TODO: check this works
    for (long i = 0; i < *mat_m; i++) {
        free(mat[i]);
    }
    free(mat);

    return trans;
}

void _print_matrix(double **mat, long mat_m, long mat_n) {
    for (long i = 0; i < mat_m; i++) {
        for (long j = 0; j < mat_n; j++) {
            printf("%.2lf ", mat[i][j]);
        }
        printf("\n");
    }
}


void free_matrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


int main(int argc, char *argv[]) {
    if (argc < 5) {
        perror("Needed dimensions of both matrices.\n");
        return EXIT_FAILURE;
    }

    long a_m = atol(argv[1]);
    long a_n = atol(argv[2]);
    long b_m = atol(argv[3]);
    long b_n = atol(argv[4]);

    if (a_n != b_m) {
        perror("a_n and b_m must be equal\n");
        return EXIT_FAILURE;
    }

    double **mat_a = NULL;
    double **mat_b = NULL;
    double **res = NULL;
    long n_rows = 0;    // num of rows of mat a that each node has

    int node = 0, npes;
    struct timeval t_prev, t_init, t_final;
    double overhead, total_time;

    gettimeofday(&t_prev,NULL);
    gettimeofday(&t_init,NULL);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);

    	
    if (npes - 1 > b_m) {
        //printf("Número de nodos excesivo para el tamaño de la matriz.\n");
        //printf("Reduzca el número de nodos.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (!node) {
        srand(time(NULL));
        
        mat_a = _gen_matrix(a_m, a_n);
        mat_b = _gen_matrix(b_m, b_n);

        n_rows = a_m / (npes - 1);  // ten en conta que desta forma o 0 traballa cando a_m % npes != 0
    }

    MPI_Bcast(&n_rows, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&a_n, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b_m, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b_n, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    short flag;
    if (!node) {
        MPI_Recv(&flag, 1, MPI_SHORT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
        for (int dest = 1; dest < npes; dest++) {
            for (long i = 0; i < b_m; i++) {
                MPI_Send(mat_b[i], b_n, MPI_DOUBLE, dest, dest, MPI_COMM_WORLD);
            }
        }
    } else {
        
        mat_b = _alloc_matrix(b_m, b_n);

        if (node == 1) {    // tells node 0 to start working
            MPI_Send(&flag, 1, MPI_SHORT, 0, 1, MPI_COMM_WORLD);
        }

        for (long i = 0; i < b_m; i++) {
            MPI_Recv(mat_b[i], b_n, MPI_DOUBLE, 0, node, MPI_COMM_WORLD, NULL);
        }
    }

    // sincronizamos todolos procesos, para pasar as filas que corresponden
    MPI_Barrier(MPI_COMM_WORLD);

    if (!node) {
        MPI_Recv(&flag, 1, MPI_SHORT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
        long fila = 0;
        for (int dest = 1; dest < npes; dest++) {
            for (long i = 0; i < n_rows; i++) {
                fila = (dest - 1) * n_rows + i;
                MPI_Send(mat_a[fila], a_n, MPI_DOUBLE, dest, dest, MPI_COMM_WORLD);
            }
        }
    } else {
        a_m = n_rows;
        mat_a = _alloc_matrix(a_m, a_n);

        if (node == 1) {    // tells node 0 to start working
            MPI_Send(&flag, 1, MPI_SHORT, 0, 1, MPI_COMM_WORLD);
        }

        for (long i = 0; i < a_m; i++) {
            MPI_Recv(mat_a[i], a_n, MPI_DOUBLE, 0, node, MPI_COMM_WORLD, NULL);
        }
    }

    res = _alloc_matrix(a_m, b_n);

    int start = 0;
    if (!node) {
        if (n_rows * (npes - 1) < a_m) {
            start = (npes - 1) * n_rows;
            for (long i = start; i < a_m; i++) {
                for (long j = 0; j < b_n; j++) {
                    res[i][j] = 0;
                    for (long k = 0; k < a_n; k++) {
                        res[i][j] += mat_a[i][k] * mat_b[k][j];
                    }
                }
            }
            printf("ENTRE Y TODO BIEN\n");
        }
        
    } else {
        for (long i = 0; i < a_m; i++) {
            for (long j = 0; j < b_n; j++) {
                res[i][j] = 0;
                for (long k = 0; k < a_n; k++) {
                    res[i][j] += mat_a[i][k] * mat_b[k][j];
                }
            }
        }
    }

    // sincronizamos todolos procesos, para pasar os resultados a 0
    MPI_Barrier(MPI_COMM_WORLD);

    if (!node) {

        for (int dest = 1; dest < npes; dest++) {
            MPI_Send(&flag, 1, MPI_SHORT, dest, dest, MPI_COMM_WORLD);
            for (long i = 0; i < n_rows; i++) {
                // é dest-1 porque eu os que calculo na 3P son os do final
                MPI_Recv(&res[(dest - 1)*n_rows + i], b_n, MPI_DOUBLE, dest, dest, MPI_COMM_WORLD, NULL);
            }
        }

    } else {
        MPI_Recv(&flag, 1, MPI_SHORT, 0, node, MPI_COMM_WORLD, NULL);
        for (long i = 0; i < a_m; i++) {
            MPI_Send(res[i], b_n, MPI_DOUBLE, 0, node, MPI_COMM_WORLD);
        }
    }

    gettimeofday(&t_final,NULL);
    overhead = (t_init.tv_sec-t_prev.tv_sec+(t_init.tv_usec-t_prev.tv_usec)/1.e6);
    total_time = (t_final.tv_sec-t_init.tv_sec+(t_final.tv_usec-t_init.tv_usec)/1.e6)-overhead;

    if (!node) {
        printf("\nRESULT:\n");
        _print_matrix(res, a_m, b_n);
        printf("\n");

        FILE *fp = fopen("res_2P.csv", "a");
        if (!fp) {
            printf("Error opening CSV\n");
            return EXIT_FAILURE;
        }
        
        fprintf(fp, "%d, %ld, %ld, %ld, %ld, %.50f\n", npes, a_m, a_n, b_m, b_n, total_time);
        // Closing CSV file
        fclose(fp); 
    }

    free_matrix(mat_a, a_m);
    free_matrix(mat_b, b_m);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
