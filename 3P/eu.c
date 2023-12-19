#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <sys/time.h>

#define MAX 100000

double * receive_send(int node);

double ** _get_matrix(FILE *fp, int *m, int *n) {
    double **matrix;
    char line[MAX];
    // format for first line is [INT]x[INT]
    if (!fgets(line, MAX, fp)) {
        printf("Error reading file\n");
        exit(1);
    }
    *n = atoi(strchr(line, 'x') + 1*sizeof(char));
    char dst_m[MAX];
    strncpy(dst_m, line, strchr(line, 'x') - line); // size is difference of pointer locations
    *m = atoi(dst_m);

    if (!(matrix = malloc(sizeof(double*)*(*m)))) {
        printf("Error allocating for matrix\n");
        exit(1);
    }
    for (int i = 0; i < *m; i++) {
        if (!(matrix[i] = malloc(sizeof(double)*(*n)))) {
            printf("Error allocating for matrix\n");
            exit(1);
        }
    }

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

double * _get_vector(FILE *fp, int *length) {
    double *vector;
    char line[MAX];
    // format for first line is [INT]x[INT]
    if (!fgets(line, MAX, fp)) {
        printf("Error reading file\n");
        exit(1);
    }
    int n = atoi(strchr(line, 'x') + 1*sizeof(char));
    char dst_m[MAX];
    strncpy(dst_m, line, strchr(line, 'x') - line); // size is difference of pointer locations
    int m = atoi(dst_m);

    *length = (m == 1) ? n : m;

    if (!(vector = malloc(sizeof(double)*(*length)))) {
        printf("Error allocating for vector\n");
        exit(1);
    }

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

double ** _transpose(double **mat, int *mat_m, int *mat_n) {
    double **trans;
    
    if (!(trans = malloc(sizeof(double*)*(*mat_n)))) {
        printf("Error allocating transpose\n");
        exit(1);
    }

    for (int i = 0; i < *mat_n; i++) {
        if (!(trans[i] = malloc(sizeof(double)*(*mat_m)))) {
            printf("Error allocating transpose\n");
            exit(1);
        }
    }

    for (int i = 0; i < *mat_n; i++) {
        for (int j = 0; j < *mat_m; j++) {
            trans[i][j] = mat[j][i];
        }
    }
    
    int temp = *mat_m;
    *mat_m = *mat_n;
    *mat_n = temp;

    return trans;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {     // optional: create random matrices
        printf("Needed the direction of the multiplication (0 = matrix*vector, 1 = vector*matrix),\n");
        printf("a file with a matrix and one with a vector to be passed\n");
        printf("Both with the next format:\n");
        printf("SIZExSIZE\nx00, x01, ..., x0n\n..., ..., ..., ...\nxm0, xm1, ..., xmn\n");
        return 1;
    }

    int dir = atoi(argv[1]);
    FILE *fp = fopen(argv[2], "r");
    FILE *fq = fopen(argv[3], "r");
    if (!fp || !fq) {
        printf("Error opening files\n");
        return 1;
    }

    int node = 0, npes;
    struct timeval t_prev, t_init, t_final;
    double overhead, total_time;

    int matrix_m = 0;
    int matrix_n = 0;
    double **matrix = NULL;
    int vector_l = 0;
    double *vector = NULL;
    int n_cols = 0;
    double *result = NULL;
    double *res_n = NULL;

    gettimeofday(&t_prev, NULL);
    gettimeofday(&t_init, NULL);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);

    if (!node) {    // node 0 builds matrices
        matrix = _get_matrix(fp, &matrix_m, &matrix_n);
        if (dir) matrix = _transpose(matrix, &matrix_m, &matrix_n);
        vector = _get_vector(fq, &vector_l);
        fclose(fp);
        fclose(fq);

        if (!dir && matrix_n != vector_l) {
            printf("Matrix has dimension %dx%d.\n", matrix_m, matrix_n);
            printf("Vector has length %d\n", vector_l);
            printf("Unable to multiply them\n");
            return 1;
        } else if (dir && vector_l != matrix_m) {
            printf("Vector has length %d\n", vector_l);
            printf("Matrix has dimension %dx%d.\n", matrix_m, matrix_n);
            printf("Unable to multiply them\n");
            return 1;
        }

        // first broadcast vector size
        // then matrix size
        // then vector && matrix
        if (npes != 1) {
            MPI_Bcast(&vector_l, 1, MPI_INT, node, MPI_COMM_WORLD);  // vector size
            MPI_Bcast(vector, vector_l, MPI_DOUBLE, node, MPI_COMM_WORLD);
            int start_col = 0;
            n_cols = matrix_n;
            MPI_Bcast(&matrix_m, 1, MPI_INT, node, MPI_COMM_WORLD);
            MPI_Bcast(&n_cols, 1, MPI_INT, node, MPI_COMM_WORLD);
            int pos = 0;
            for (int dest = 1; dest < npes; dest++) {
                for (int i = 0; i < n_cols; i++) {
                    pos = ((dest - 1) * n_cols) + i;
                    MPI_Send(matrix[pos], matrix_m, MPI_DOUBLE, dest, dest, MPI_COMM_WORLD);  // maybe we can use tag to later reconstruct matrix
                }
            }
        }

        if (!(result = malloc(sizeof(double)*matrix_m))) {
            printf("%d: Error allocating for result\n", node);
            exit(1);
        }

        if (matrix_n % npes || npes == 1) {
            int start = (npes - 1) * n_cols;
            for (int i = start; i < matrix_m; i++) {
                for (int j = 0; j < matrix_n; j++) {
                    result[i] += vector[j] * matrix[(npes - 1) * n_cols + i][j];
                }
            }
        }
    } else {
        // 5 receives
        // 1 vector length
        // 1 vector
        // 2 matrix size
        // n_cols matrix
        res_n = receive_send(node);
    }

    if (npes != 1) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (!node) {
            char ok = '0';
            double *temp;
            if (!(temp = malloc(sizeof(double)*(matrix_n/(npes - 1))))) {
                printf("%d: Error allocating for temp\n", node);
                exit(1);
            }
            for (int i = 1; i < npes; i++) {
                MPI_Send(&ok, 1, MPI_CHAR, i, i, MPI_COMM_WORLD);
                MPI_Recv(temp, n_cols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, NULL);
                for (int j = 0; j < n_cols; j++) {
                    result[(i - 1) * n_cols + j] = temp[j];
                }
            }
        } else {
            char ok;
            MPI_Recv(&ok, 1, MPI_CHAR, 0, node, MPI_COMM_WORLD, NULL);
            MPI_Send(res_n, n_cols, MPI_DOUBLE, 0, node, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < vector_l; i++) {
        if (!dir) printf("%lf\n", result[i]);
        else printf("%lf ", result[i]);
    }
    if (dir) printf("\n");

    gettimeofday(&t_final, NULL);
    overhead = (t_init.tv_sec - t_prev.tv_sec + (t_init.tv_usec - t_prev.tv_usec)/1.e6);
    total_time = (t_final.tv_sec - t_init.tv_sec + (t_final.tv_usec - t_init.tv_usec)/1.e6) - overhead;

    return EXIT_SUCCESS;
}

double * receive_send(int node) {
    int vec_l;
    double *vec;
    int mat_m, mat_n;
    double **mat;

    MPI_Recv(&vec_l, 1, MPI_INT, 0, node, MPI_COMM_WORLD, NULL);
    if (!(vec = malloc(sizeof(double)*vec_l))) {
        printf("Error allocating for vector\n");
        exit(1);
    }
    MPI_Recv(vec, vec_l, MPI_DOUBLE, 0, node, MPI_COMM_WORLD, NULL);
    MPI_Recv(&mat_m, 1, MPI_INT, 0, node, MPI_COMM_WORLD, NULL);
    MPI_Recv(&mat_n, 1, MPI_INT, 0, node, MPI_COMM_WORLD, NULL);

    if (!(mat = malloc(sizeof(double*)*mat_m))) {
        printf("Error allocating for matrix\n");
        exit(1);
    }

    for (int i = 0; i < mat_m; i++) {
        if (!(mat[i] = malloc(sizeof(double)*mat_n))) {
            printf("Error allocating for matrix\n");
            exit(1);
        }
    }

    for (int i = 0; i < mat_m; i++) {
        MPI_Recv(mat, mat_n, MPI_DOUBLE, 0, node, MPI_COMM_WORLD, NULL);
    }

    // mult
    double *result;
    if (!(result = malloc(sizeof(double)*mat_m))) {
        printf("Error allocating for result\n");
        exit(1);
    }

    for (int i = 0; i < mat_m; i++) {
        for (int j = 0; j < mat_n; j++) {
            result[i] += vec[j] * mat[i][j];
        }
    }

    return result;
}
