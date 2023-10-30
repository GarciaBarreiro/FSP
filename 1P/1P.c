#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>
#include <locale.h>

// absolute substraction
double abs_subs(double a, double b) {
    return a > b ? a - b : b - a;
}

double gaussian(double x) {
    return exp(-pow(x, 2));
}

// calculates integral by using trapezoids
// and calculates trapezoids by calculating a square and adding/substracting a triangle
double trapezoids(double x, double step) {
    double y1 = gaussian(x);
    double y2 = gaussian(x + step);

    double square_area = y1 * step;
    double triangle_area = abs_subs(y1, y2) * step / 2;
    return y1 > y2 ? square_area - triangle_area : square_area + triangle_area;
}

int main(int argc, char *argv[]) {
    int node = 0, npes;
    int neg_x = -10, pos_x = 10;        // range of function to calculate (from x -10 to x 10)
    long int num_iter = 100000000;  // number of iterations each node does
    double reference = 3.1415926535897932384626433832795028841971693993751058209749446;

    struct timeval previa, inicio, final;
    double overhead,total_time;
    
    gettimeofday(&previa,NULL);
    gettimeofday(&inicio,NULL);
    
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
    }

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
            // printf("node %d, receiving from node %d\n", node, node+jump);
            total += msg;
        } else if (node - jump >= 0 && node < tot_nodes) {
            sleep(1);
            MPI_Send(&total, 1, MPI_DOUBLE, node - jump, 0, MPI_COMM_WORLD);
            // printf("node %d, sending to node %d\n", node, node-jump);
        }

        tot_nodes = node_step;
    }

    gettimeofday(&final,NULL);
    overhead = (inicio.tv_sec-previa.tv_sec+(inicio.tv_usec-previa.tv_usec)/1.e6);
    total_time = (final.tv_sec-inicio.tv_sec+(final.tv_usec-inicio.tv_usec)/1.e6)-overhead;

    if (!node){     // i think we can do this outside MPI_Finalize                 
        double pi = total*total;
        double error = abs_subs(pi,reference);
        double quality = 1/(error*total_time);

        printf("---------------------- RESULTS -----------------------\n");
        printf("PI == %.50f\n",pi);
        printf("Difference btwn reference == %.50f\n",error);
        printf("Time == %.50f\n", total_time);
        printf("Quality == %.50f\n",quality);
        printf("------------------------------------------------------\n");
        
        // Create CSV file
        FILE* fp = fopen("results.csv", "a");
        if (fp == NULL) {
            printf("Error opening file %s\n", "results.csv");
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
