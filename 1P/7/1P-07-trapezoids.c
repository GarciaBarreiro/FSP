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

double func(double x) {
    return (pow(x,4)*pow(1-x,4))/pow(1+x,2);
}

// calculates integral by using trapezoids
// and calculates trapezoids by calculating a square and adding/substracting a triangle
double trapezoids(double x, double step) {
    double y1 = func(x);
    double y2 = func(x + step);

    double square_area = y1 * step;
    double triangle_area = abs_subs(y1, y2) * step / 2;
    //printf("square area = %lf, triangle_area = %lf\n", square_area, triangle_area);
    return y1 > y2 ? square_area - triangle_area : square_area + triangle_area;
}

int main(int argc, char *argv[]) {
    int node = 0, npes;
    double neg_x = 0,       // a movida é que con 3 dá NaN (división por 0)
           pos_x = 1;       // range of function to calculate (from x -3 to x 3)
    long int num_iter = 1000000;      // number of iterations each node does
    // long int num_iter = 10000000;      // number of iterations each node does
    double reference = 3.1415926535897932384626433832795028841971693993751058209749446;

    struct timeval t_prev, t_init, t_final;
    struct timespec sleep_time = {0,1000000};   // Sleep for 1ms
    double overhead,total_time;
    
    gettimeofday(&t_prev,NULL);
    gettimeofday(&t_init,NULL);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    double size = (pos_x - neg_x) / (double)npes;   // how many units each node is going to calculate
    MPI_Comm_rank(MPI_COMM_WORLD, &node);

    // ------------------------------
    //          pi calculus
    // ------------------------------

    double start_x = neg_x + size*node;     // starting position for each node
    double step = size/num_iter;            // how much will x increase each iteration
    double total = 0;
    for (int i = 0; i < num_iter; i++) {
        total += trapezoids(start_x, step);
        start_x += step;
        if (!(i % 1000000)) printf("total = %lf\n", total);
    }
    printf("total = %lf\n", total);
    printf("pi = %lf\n", total/3);

    // ------------------------------
    //         message passing
    // ------------------------------

    // basically the first n/2 nodes receive and the others send
    // when n is odd, the node in the middle does nothing
    // next iteration, only the first n/2 nodes are taken into account
    // the first half receives, the second half sends
    // etc

    double msg;
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
            MPI_Recv(&msg, 1, MPI_DOUBLE, node + jump, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
            total += msg;
        } else if (node - jump >= 0 && node < tot_nodes) {
            nanosleep(&sleep_time,&sleep_time);
            MPI_Send(&total, 1, MPI_DOUBLE, node - jump, 0, MPI_COMM_WORLD);
        }

        tot_nodes = node_step;
    }

    gettimeofday(&t_final,NULL);
    overhead = (t_init.tv_sec-t_prev.tv_sec+(t_init.tv_usec-t_prev.tv_usec)/1.e6);
    total_time = (t_final.tv_sec-t_init.tv_sec+(t_final.tv_usec-t_init.tv_usec)/1.e6)-overhead;

    if (!node){
        double pi = 22.0/7 - total;
        double error = abs_subs(pi,reference);
        double quality = 1/(error*total_time);

        printf("---------------------- RESULTS -----------------------\n");
        printf("PI == %.50f\n",pi);
        printf("Difference btwn reference == %.50f\n",error);
        printf("Time == %.50f\n", total_time);
        printf("Quality == %.50f\n",quality);
        printf("------------------------------------------------------\n");
        
        // Create CSV file
        FILE* fp = fopen("results_trapezoids.csv", "a");
        if (fp == NULL) {
            printf("Error opening file results_trapezoids.csv\n");
            exit(EXIT_FAILURE);
        }
        setlocale(LC_ALL, "es_ES.utf8");
        fprintf(fp,"%d;%.50f;%.50f;%-50f\n", npes, total_time, error, quality);
        // Closing CSV file
        fclose(fp);

    }
    MPI_Finalize();

    return EXIT_SUCCESS;
}

