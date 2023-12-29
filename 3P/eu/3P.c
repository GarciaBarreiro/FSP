#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// this is for reading matrices from files, functions not used
#define MAX 1000000

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

double *_alloc_vector(long vec_l) {
    double *vector;

    if (!(vector = malloc(sizeof(double)*vec_l))) {
        printf("Error allocating vector\n");
        exit(1);
    }

    return vector;
}

// reads matrix from file
// UNUSED
// unimplemented: read by columns when dir != 0
double ** _read_matrix(FILE *fp, long *m, long *n) {
    char line[MAX];
    // format for first line is [INT]x[INT]
    if (!fgets(line, MAX, fp)) {
        printf("Error reading file\n");
        exit(1);
    }
    *n = atol(strchr(line, 'x') + 1*sizeof(char));
    char dst_m[MAX];
    strncpy(dst_m, line, strchr(line, 'x') - line); // size is difference of pointer locations
    *m = atol(dst_m);

    double **matrix = _alloc_matrix(*m, *n);

    for (int i = 0; i < *m; i++) {
        if (!fgets(line, MAX, fp)) {
            printf("Error reading file\n");
            exit(1);
        }
        char *rest = line;
        matrix[i][0] = atof(strtok_r(rest, " ", &rest));
        for (int j = 1; j < *n; j++) {
            matrix[i][j] = atof(strtok_r(rest, " ", &rest));
        }
    }
    return matrix;
}

// reads vector from file
// UNUSED
double * _read_vector(FILE *fp, long *length) {
    char line[MAX];
    // format for first line is [INT]x[INT]
    if (!fgets(line, MAX, fp)) {
        printf("Error reading file\n");
        exit(1);
    }
    long n = atol(strchr(line, 'x') + 1*sizeof(char));
    char dst_m[MAX];
    strncpy(dst_m, line, strchr(line, 'x') - line); // size is difference of pointer locations
    long m = atol(dst_m);

    *length = (m == 1) ? n : m;

    double *vector = _alloc_vector(*length);

    if (m == 1) {       // horizontal vector
        if (!fgets(line, MAX, fp)) {
            printf("Error reading file\n");
            exit(1);
        }

        char *rest = line;
        vector[0] = atof(strtok_r(rest, " ", &rest));
        
        for (int i = 1; i < *length; i++) {
            vector[i] = atof(strtok_r(rest, " ", &rest));
        }
    } else {            // vertical vector
        for (int i = 0; i < *length; i++) {
            if (!fgets(line, MAX, fp)) {
                printf("Error reading file\n");
                exit(1);
            }

            vector[i] = atof(line);
        }
    }

    return vector;
}

double **_gen_matrix(long mat_m, long mat_n) {
    double **matrix = _alloc_matrix(mat_m, mat_n);

    for (int i = 0; i < mat_m; i++) {
        for (int j = 0; j < mat_n; j++) {
            matrix[i][j] = rand() % 1000 + ((rand() % 100) / 100.0);
        }
    }

    return matrix;
}

double *_gen_vector(long vec_l) {
    double *vector = _alloc_vector(vec_l);

    for (long i = 0; i < vec_l; i++) {
        vector[i] = rand() % 1000 + ((rand() % 100) / 100.0);
    }

    return vector;
}

double **_transpose(double **mat, long *mat_m, long *mat_n) {
    double **trans = _alloc_matrix(*mat_n, *mat_m);

    for (long i = 0; i < *mat_n; i++) {
        for (long j = 0; j < *mat_m; j++) {
            trans[i][j] = mat[j][i];
        }
    }
    
    int temp = *mat_m;
    *mat_m = *mat_n;
    *mat_n = temp;

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

void _print_mult(int dir, double **mat, long mat_m, long mat_n, double *vec, long vec_l) {
    if (!dir) {
        printf("Matrix:\n");
        _print_matrix(mat, mat_m, mat_n);
        printf("\nVector:\n");
        for (long i; i < vec_l; i++) {
            printf("%.2lf\n", vec[i]);
        }
        printf("\n");
    } else {
        printf("Vector:\n");
        for (long i; i < vec_l; i++) {
            printf("%.2lf ", vec[i]);
        }
        printf("\n\nMatrix:\n");
        _print_matrix(mat, mat_m, mat_n);
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

    // these variables will be needed by all nodes
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

        if (mat_m < 10 && mat_n < 10) _print_mult(dir, mat, mat_m, mat_n, vec, vec_l);

        if (dir) mat = _transpose(mat, &mat_m, &mat_n);
        vec = _gen_vector(vec_l);

        n_rows = mat_m / (npes - 1);
    }

    MPI_Bcast(&vec_l, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (node) {
        vec = _alloc_vector(vec_l);
    }

    MPI_Bcast(vec, vec_l, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&mat_m, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_rows, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    short flag = 0;
    if (!node) {
        // waits until node 1 finishes allocating, then starts sending data
        MPI_Recv(&flag, 1, MPI_SHORT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
        long col = 0;
        for (int dest = 1; dest < npes; dest++) {
            for (int i = 0; i < n_rows; i++) {
                col = (dest - 1) * n_rows + i;
                MPI_Send(mat[col], mat_n, MPI_DOUBLE, dest, dest, MPI_COMM_WORLD);
            }
        }
    } else {
        mat_m = n_rows;
        mat = _alloc_matrix(mat_m, mat_n);

        if (node == 1) {
            MPI_Send(&flag, 1, MPI_SHORT, 0, 1, MPI_COMM_WORLD);
        }

        for (long i = 0; i < mat_m; i++) {
            MPI_Recv(mat[i], mat_n, MPI_DOUBLE, 0, node, MPI_COMM_WORLD, NULL);
        }
    }

    double *res = _alloc_vector(mat_m);

    int start = 0;
    if (!node) {
        if (n_rows * (npes - 1) < mat_m) {
            start = (npes - 1) * n_rows;
            for (long i = start; i < mat_m; i++) {
                for (long j = 0; j < mat_n; j++) {
                    res[i] += vec[j] * mat[i][j];
                }
            }
        }
    } else {
        for (long i = 0; i < mat_m; i++) {
            for (long j = 0; j < mat_n; j++) {
                res[i] += vec[j] * mat[i][j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (!node) {
        for (int dest = 1; dest < npes; dest++) {
            MPI_Send(&flag, 1, MPI_SHORT, dest, dest, MPI_COMM_WORLD);
            MPI_Recv(&res[(dest - 1)*n_rows], n_rows, MPI_DOUBLE, dest, dest, MPI_COMM_WORLD, NULL);
        }
    } else {
        MPI_Recv(&flag, 1, MPI_SHORT, 0, node, MPI_COMM_WORLD, NULL);
        MPI_Send(res, mat_m, MPI_DOUBLE, 0, node, MPI_COMM_WORLD);
    }

    gettimeofday(&t_final, NULL);
    overhead = (t_init.tv_sec-t_prev.tv_sec+(t_init.tv_usec-t_prev.tv_usec)/1.e6);
    total_time = (t_final.tv_sec-t_init.tv_sec+(t_final.tv_usec-t_init.tv_usec)/1.e6)-overhead;

    if (!node) {
        printf("\n\nRESULT:\n");
        for (long i = 0; i < mat_m; i++) {
            printf("%.4lf%s", res[i], dir ? " " : "\n");
        }
        printf("\n");

        FILE *fp = fopen("res_3P_6.csv", "a");
        if (!fp) {
            printf("Error opening res file\n");
            return 1;
        }

        fprintf(fp, "%d, %d, %ld, %ld, %ld, %.50f\n", npes, dir, mat_m, mat_n, vec_l, total_time);
    }

    MPI_Finalize();
}
