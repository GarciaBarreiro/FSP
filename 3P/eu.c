#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <sys/time.h>

#define MAX 100000

double ** parse_file(FILE *fp, int *n, int *m) {
    double **matrix;
    char line[MAX];
    // format for first line is [INT]x[INT]
    if (!fgets(line, MAX, fp)) {
        printf("Error reading file\n");
        exit(1);
    }
    *m = atoi(strchr(line, 'x') + 1*sizeof(char));
    char dst_n[MAX];
    strncpy(dst_n, line, strchr(line, 'x') - line); // size is difference of pointer locations
    *n = atoi(dst_n);
    printf("n == %d, m == %d\n", *n, *m);

    if (!(matrix = malloc(sizeof(double*)*(*n)))) {
        printf("Error allocating for matrix\n");
        exit(1);
    }
    for (int i = 0; i < *n; i++) {
        if (!(matrix[i] = malloc(sizeof(double)*(*m)))) {
            printf("Error allocating for matrix\n");
            exit(1);
        }
    }

    for (int i = 0; i < *n; i++) {
        printf("%d\n", i);
        if (!fgets(line, MAX, fp)) {
            printf("Error reading file\n");
            exit(1);
        }
        char *rest = line;
        matrix[i][0] = atof(strtok_r(rest, " ", &rest));
        printf("%lf\n", matrix[i][0]);
        for (int j = 1; j < *m; j++) {
            matrix[i][j] = atof(strtok_r(rest, " ", &rest));
            printf("%lf\n", matrix[i][j]);
        }
    }
    return matrix;
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

    int matrix_n, matrix_m;
    double **matrix = parse_file(fp, &matrix_n, &matrix_m);
    fclose(fp);
    int vector_n, vector_m; // a vector is just a 1xN (or Nx1) matrix
    double **vector = parse_file(fq, &vector_n, &vector_m);
    fclose(fq);

    if (!dir && vector_m != matrix_n) {
        printf("Vector has dimension %dx%d and\n", vector_n, vector_m);
        printf("matrix has dimension %dx%d.\n", matrix_n, matrix_m);
        return 1;
    } else if (dir && matrix_m != vector_n) {
        printf("Matrix has dimension %dx%d and\n", matrix_n, matrix_m);
        printf("vector has dimension %dx%d.\n", vector_n, vector_m);
        return 1;
    }

    printf("hello\n");
    for (int i = 0; i < matrix_n; i++) {
        for (int j = 0; j < matrix_m; j++) {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }

    int node = 0, npes;
    struct timeval t_prev, t_init, t_final;
    double overhead, total_time;

    gettimeofday(&t_prev, NULL);
    gettimeofday(&t_init, NULL);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);

    gettimeofday(&t_final, NULL);
    overhead = (t_init.tv_sec - t_prev.tv_sec + (t_init.tv_usec - t_prev.tv_usec)/1.e6);
    total_time = (t_final.tv_sec - t_init.tv_sec + (t_final.tv_usec - t_init.tv_usec)/1.e6) - overhead;

    return EXIT_SUCCESS;
}
