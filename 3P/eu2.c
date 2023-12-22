#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>    // test if this works in cesga
#include <string.h>
#include <time.h>
#include <sys/time.h>

// this is for reading matrices from files, not implemented here
#define MAX 1000000

double **_gen_matrix(int mat_m, int mat_n) {
    double **matrix;

    if (!(matrix = malloc(sizeof(double*)*mat_m))) {
        printf("Error allocating matrix\n");
        exit(1);
    }
    for (int i = 0; i < mat_m; i++) {
        if (!(matrix[i] = malloc(sizeof(double)*mat_n))) {
            printf("Error allocating matrix\n");
            exit(1);
        }
    }

    for (int i = 0; i < mat_m; i++) {
        for (int j = 0; j < mat_n; j++) {
            matrix[i][j] = rand();
        }
    }
    return matrix;
}

double *_gen_vector(int vec_l) {
    double *vector;

    if (!(vector = malloc(sizeof(double)*vec_l))) {
        printf("Error allocating for vector\n");
        exit(1);
    }

    for (int i = 0; i < vec_l; i++) {
        vector[i] = rand();
    }

    return vector;
}

double **_transpose(double ***mat, long *mat_m, long *mat_n) {
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
            trans[i][j] = *mat[j][i];
        }
    }
    
    int temp = *mat_m;
    *mat_m = *mat_n;
    *mat_n = temp;

    // TODO: check this works
    for (long i = 0; i < *mat_m; i++) {
        free(*mat[i]);
    }
    free(mat);

    return trans;
}

void _print_matrix(double **mat, int mat_m, int mat_n) {
    for (int i = 0; i < mat_m; i++) {
        for (int j = 0; j < mat_n; j++) {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Needed the direction of the multiplication (0 = matrix x vector, 1 = vector x matrix),\n");
        printf("the two dimensions of a matrix and the length of a vector\n");
        return 1;
    }
    
    int dir = atoi(argv[1]);
    long mat_m = atol(argv[2]);
    long mat_n = atol(argv[3]);
    long vec_l = atol(argv[4]);

    if (!dir && mat_n != vec_l) {
        printf("Matrix has dimension %ldx%ld.\n", mat_m, mat_n);
        printf("Vector has length %ld\n", vec_l);
        printf("Unable to multiply them\n");
        return 1;
    } else if (dir && vec_l != mat_m) {
        printf("Vector has length %ld\n", vec_l);
        printf("Matrix has dimension %ldx%ld.\n", mat_m, mat_n);
        printf("Unable to multiply them\n");
        return 1;
    }

    // all vars that rank 0 will have fully
    double **mat = NULL;
    double *vec = NULL;
    long n_rows = 0;
    double *result = NULL;
    double *res_n = NULL;

    int node = 0, npes;
    struct timeval t_prev, t_init, t_final;
    double overhead, total_time;

    gettimeofday(&t_prev, NULL);
    gettimeofday(&t_init, NULL);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);

    if (npes == 1) {
        printf("This program only works with more than one node,\n");
        printf("consider adding at least another one.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (npes > mat_n) {
        printf("There are more nodes than %s,\n", dir ? "rows" : "columns");
        printf("consider running with less nodes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (!node) {
        srand(time(NULL));
        mat = _gen_matrix(mat_m, mat_n);
        if (dir) mat = _transpose(&mat, &mat_m, &mat_n);
        vec = _gen_vector(vec_l);
        n_rows = mat_m / npes;
    }

    MPI_Bcast(&vec_l, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (node) {
        if (!(vec = malloc(sizeof(double)*vec_l))) {
            printf("%d: Error allocating vector\n", node);
            return 1;
        }
    }

    MPI_Bcast(&vec, vec_l, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&mat_m, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_rows, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (!node) {
        // waits until node 1 finishes allocating, then starts sending data
        MPI_Recv(NULL, 1, MPI_SHORT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
        long col = 0;
        for (int dest = 1; dest < npes + 1; dest++) {
            for (int i = 0; i < n_rows; i++) {
                col = (dest - 1) * n_rows + i;
                MPI_Send(mat[col], mat_n, MPI_DOUBLE, dest, dest, MPI_COMM_WORLD);
            }
        }
    } else {
        mat_m = n_rows;
        if (!(mat = malloc(sizeof(double*)*mat_m))) {
            printf("%d: Error allocating matrix\n", node);
            return 1;
        }

        for (long i = 0; i < mat_m; i++) {
            if (!(mat[i] = malloc(sizeof(double)*mat_n))) {
                printf("%d: Error allocating matrix\n", node);
                exit(1);
            }
        }

        if (node == 1) {
            MPI_Send(NULL, 1, MPI_SHORT, 0, 1, MPI_COMM_WORLD);
        }

        for (long i = 0; i < mat_m; i++) {
            MPI_Recv(mat[i], mat_n, MPI_DOUBLE, 0, node, MPI_COMM_WORLD, NULL);
        }
    }

    printf("rank: %d\n", node);
    _print_matrix(mat, mat_m, mat_n);

    // TODO: multiplication
}
