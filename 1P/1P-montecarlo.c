#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <locale.h>

// absolute substraction
double abs_subs(double a, double b) {
    return a > b ? a - b : b - a;
}

double gaussian(double x) {
    return exp(-pow(x, 2));
}

// calculates integral using monte carlo
long int montecarlo(long int num_iter, int x_range, int y_range) {
    double x, y;
    long int total = 0;

    for (int i = 0; i < num_iter; i++) {
        x = (rand() % (x_range * 10000) - x_range*5000)/(double)10000;
        y = (rand() % (y_range * 10000))/(double)10000;

        if (y < gaussian(x)) total++;
    }
    return total;
}

int main(int argc, char *argv[]) {
    int node = 0, npes;
    int x_range = 10, y_range = 10; // x is [-5,5]; y is [0,10]
    long int num_iter = argc > 1 ? atoi(argv[1]) : 100000000;      // number of iterations each node does
    double reference = 3.1415926535897932384626433832795028841971693993751058209749446;

    struct timeval t_prev, t_init, t_final;
    struct timespec sleep_time = {0,1000000};   // Sleep for 1ms
    double overhead,total_time;
    
    gettimeofday(&t_prev,NULL);
    gettimeofday(&t_init,NULL);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &node);

    // ------------------------------
    //          pi calculus
    // ------------------------------

    srand(time(NULL)/(node+1));
    long int total = montecarlo(num_iter, x_range, y_range);

    // ------------------------------
    //         message passing
    // ------------------------------

    // basically the first n/2 nodes receive and the others send
    // when n is odd, the node in the middle does nothing
    // next iteration, only the first n/2 nodes are taken into account
    // the first half receives, the second half sends
    // etc

    long int msg;
    int node_step;
    int iter = npes % 2 ? npes/2 + 1 : npes/2;  // number of iterations. if npes is odd, one more than half
    int tot_nodes = npes;                       // node total each iter
    int jump;
    int flag = 0;

    for (int i = 0; i < iter; i++) {
        msg = 0;
        node_step = flag ? tot_nodes / 2 + 1 : tot_nodes / 2;
        jump = node_step;

        if (flag) flag = 0;

        if (node_step % 2 && tot_nodes % 2) {
            jump++;
            flag = 1;
        }

        if (node < node_step) {
            MPI_Recv(&msg, 1, MPI_LONG, node + jump, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
            total += msg;
        } else if (node - jump >= 0 && node < tot_nodes) {
            nanosleep(&sleep_time,&sleep_time);
            MPI_Send(&total, 1, MPI_LONG, node - jump, 0, MPI_COMM_WORLD);
        }

        tot_nodes = node_step;
    }

    gettimeofday(&t_final,NULL);
    overhead = (t_init.tv_sec-t_prev.tv_sec+(t_init.tv_usec-t_prev.tv_usec)/1.e6);
    total_time = (t_final.tv_sec-t_init.tv_sec+(t_final.tv_usec-t_init.tv_usec)/1.e6)-overhead;

    if (!node){
        double root_pi = total / (double)(num_iter * npes) * 100;
        double pi = root_pi * root_pi;
        double error = abs_subs(pi,reference);
        double quality = 1/(error*total_time);

        printf("---------------------- RESULTS -----------------------\n");
        printf("PI == %.50f\n",pi);
        printf("Difference btwn reference == %.50f\n",error);
        printf("Time == %.50f\n", total_time);
        printf("Quality == %.50f\n",quality);
        printf("------------------------------------------------------\n");
        
        // Create CSV file
        FILE* fp = fopen("results_montecarlo.csv", "a");
        if (fp == NULL) {
            printf("Error opening file results.csv\n");
            exit(EXIT_FAILURE);
        }
        setlocale(LC_ALL, "es_ES.utf8");
        if (argc > 2)
            fprintf(fp,"%d;%.50f;%.50f;%.50f\n", npes, total_time, error, quality, num_iter);
        else
            fprintf(fp,"%d;%.50f;%.50f;%.50f\n", npes, total_time, error, quality);
        // Closing CSV file
        fclose(fp);

    }
    MPI_Finalize();

    return EXIT_SUCCESS;
}
