#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// absolute substraction
double abs_subs(double a, double b) {
    return a > b ? a - b : b - a;
}

double gaussian(double x) {
    return exp(-pow(x, 2));
}

double trapecios(double x, double step) {
    double y1 = gaussian(x);
    double y2 = gaussian(x + step);

    double square_area = y1 * step;
    double triangle_area = abs_subs(y1, y2) * step / 2;

    return y1 > y2 ? square_area - triangle_area : square_area + triangle_area;
}

int main(int argc, char *argv[]) {
    int node = 0, npes;
    int neg_x = -10;
    int pos_x = 10;
    int num_ex = 10000;

    printf("before init, does this print anything\n");

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    double size = (pos_x - neg_x) / (double)npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &node);

    double start_x = neg_x + size*node;
    double step = size/num_ex;
    double total = 0;
    printf("node %d, doing something before main loop\n", node);
    for (int i = 0; i < num_ex; i++) {
        total += trapecios(start_x, step);
        start_x += step;
    }
    printf("node %d, doing something after main loop\n", node);

    double msg;
    int node_step;
    int iter = npes % 2 ? npes/2 + 1 : npes/2;
    if (!node) printf("nodes == %d, iters == %d\n", npes, iter);
    int tot_nodes = npes;   // total of nodes each iter
    int jump;
    int flag = 0;
    for (int i = 0; i < iter; i++) {
        msg = 0;
        node_step = flag ? tot_nodes / 2 + 1 : tot_nodes / 2;
        jump = node_step;
        printf("node_step == %d\n", node_step);
        if (flag) flag = 0;
        if (node_step % 2 && tot_nodes % 2) {        // if (flag)???
            jump++;
            if (tot_nodes % 2) flag = 1;
        }
        if (node < node_step) {
            MPI_Recv(&msg, 1, MPI_DOUBLE, node + jump, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);
            printf("node %d, receiving from node %d\n", node, node+jump);
            total += msg;
        } else if (node - jump >= 0 && node < tot_nodes) {
            MPI_Send(&total, 1, MPI_DOUBLE, node - jump, 0, MPI_COMM_WORLD);
            printf("node %d, sending to node %d\n", node, node-jump);
        }
        tot_nodes = node_step;
    }

    if (!node) {
        printf("sqrt(pi) == %lf\n", total);
        printf("pi == %.50lf\n", total*total);
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
