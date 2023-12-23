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


int main(int argc, char *argv[]) {
    if (argc < 5) {
        perror("Needed dimensions of both matrices\n");
        return EXIT_FAILURE;
    }

    // TODO: o nº de proc debe incluírse como arg da liña de comandos

    /*
     * QUE QUEDA POR FACER?
     *
     * 1.- a multiplicación
     * 2.- como mandar b
     *
     * o resto debería estar, é moi parecido ó meu
     * se non entendes algo, non dubides en avisar, que malo será que non poda responder
     */

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

    // PARA DANI:
    // mira o meu codigo da 3, eu non executo para 1 nodo, 
    // pero basicamente por como teño eu feito
    // se queres, implementar a exec. para 1 é unha parvada
    if (!node) {
        srand(time(NULL));
        mat_a = _gen_matrix(a_m, a_n);
        mat_b = _gen_matrix(b_m, b_n);

        n_rows = a_m / (npes - 1);  // ten en conta que desta forma o 0 traballa cando a_m % npes != 0
    }

    MPI_Bcast(&n_rows, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&a_n, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b_m, 1, MPI_LONG, 0, MPI_COMM_WORLD);    // polo que eu entendo, b haberia que mandala enteira
    MPI_Bcast(&b_n, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // se b se envia enteira, facer aqui un if (node) mat_b = _alloc_matrix()
    // e despois facer un loop for i < b_m
    // e dentro dese loop un MPI_Bcast de mat_b[i]
    // que diria que e a forma mais rapida de enviar unha matriz
    // se non, como esta mat_a abaixo

    short flag;
    if (!node) {
        MPI_Recv(&flag, 1, MPI_SHORT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
        long col = 0;
        for (int dest = 1; dest < npes; dest++) {
            for (long i = 0; i < n_rows; i++) {
                col = (dest - 1) * n_rows + i;
                MPI_Send(mat_a[col], a_n, MPI_DOUBLE, dest, dest, MPI_COMM_WORLD);
            }
        }
    } else {
        a_m = n_rows;
        mat_a = _alloc_matrix(a_m, a_n);
        mat_b = _alloc_matrix(b_m, b_n);

        if (node == 1) {    // tells node 0 to start working
            MPI_Send(&flag, 1, MPI_SHORT, 0, 1, MPI_COMM_WORLD);
        }

        for (long i = 0; i < a_m; i++) {
            MPI_Recv(mat_a[i], mat_n, MPI_DOUBLE, 0, node, MPI_COMM_WORLD, NULL);
        }
        // tbh, estaba pensando, que tal vez outra forma de facer isto é poñendo o tag coma o núm de columna a enviar
        // e en vez de usar MPI_Recv (bloqueante), usar MPI_IRecv (non bloqueante)
        // por se acaso algunha cousa chunga xorde aí polo medio
        // non creo que fose o caso (a min en principio non me pasou)
        // pero quen sabe
        // así todas chegan 100%
        // (habería que mirar despois que completasen todas, pero iso pode facerse un MPI_Wait()
        // e tal vez un array de identificadores de Recv (está explicando nas manpages que estas funcs devolven un ID,
        // que despois se pode usar para cousas como o MPI_Wait)
    }

    res = _alloc_matrix(a_m, b_n);
    // multiplicación, gardar resultado na matriz res
    // (non me apetece programalo, pero é unha gilipollez)
    if (!node) {
    } else {
    }

    // sincronizamos todolos procesos, para pasar os resultados a 0
    MPI_Barrier(MPI_COMM_WORLD);

    if (!node) {
        // eu o que fixen na p3 é mandar de 0 a n unha mensaxe que n estaba agardando
        // para así recibir todo en orde
        // non sei como queres facer
        for (int dest = 1; dest < npes; dest++) {
            MPI_Send(&flag, 1, MPI_SHORT, dest, dest, MPI_COMM_WORLD);
            for (long i = 0; i < n_rows; i++) {
                // é dest-1 porque eu os que calculo na 3P son os do final
                MPI_Recv(&res[(dest - 1)*n_rows + i], n_rows, MPI_DOUBLE, dest, dest, MPI_COMM_WORLD, NULL);
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
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
