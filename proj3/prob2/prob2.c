#include <omp.h>
#include <stdio.h>

long num_steps = 10000000;
double step;

double calculate_pi(long i){
    double x = (i + 0.5) * step;
    return 4.0 / (1.0 + x * x);
}

int main(int ac, char **av){
    if (ac != 4) {
        printf("Usage: %s <schedulingType> <SIZE_CHUNK> <NUM_THREADS>\n", av[0]);
        return 1;
    }
    int type = atoi(av[1]);
    int SIZE_CHUNK = atoi(av[2]);
    int NUM_THREADS = atoi(av[3]);

    double x, pi, sum = 0.0;
    double start, end;

    start = omp_get_wtime();

    step = 1.0 / (double)num_steps;
    if (type == 1){
#pragma omp parallel for reduction(+ : sum) schedule(static, SIZE_CHUNK) num_threads(NUM_THREADS)
        for (long i = 0; i < num_steps; i++)
            sum += calculate_pi(i);
    } else if (type == 2){
#pragma omp parallel for reduction(+ : sum) schedule(dynamic, SIZE_CHUNK) num_threads(NUM_THREADS)
        for (long i = 0; i < num_steps; i++)
            sum += calculate_pi(i);
    } else if (type == 3){
#pragma omp parallel for reduction(+ : sum) schedule(guided, SIZE_CHUNK) num_threads(NUM_THREADS)
        for (long i = 0; i < num_steps; i++)
            sum += calculate_pi(i);
    } else {
        printf("Invalid scheduling type\n");
        return 1;
    }
    pi = step * sum;
    end = omp_get_wtime();

    double diff = end - start;
    printf("Execution Time : %lfms\n", diff * 1000);

    printf("pi=%.24lf\n", pi);
    return 0;
}