#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
command : 
clang -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp prob1.c -o a.out
*/
#define NUM_END 200000

int is_prime(int x) {
	if (x <= 1) 
        return 0;
	for (int i = 2; i < x; i++) {
		if (x % i == 0) 
            return 0;
	}
	return 1;
}

void count_prime(int i, int* counter) {
    if (is_prime(i)){
#pragma omp critical
		(*counter)++;
    }
}

int main(int ac, char* av[]) {
    if (ac  != 3) {
        printf("Usage: %s <schedulingType> <NUM_THREADS>\n", av[0]);
        return 1;
    }

    int NUM_THREADS = atoi(av[2]);
    int type = atoi(av[1]);
    int counter = 0;
	double t1, t2;

	t1 = omp_get_wtime();

    if (type == 1){
#pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        for (int i = 0; i < NUM_END; i++)
            count_prime(i, &counter);
    } else if (type == 2){
#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
        for (int i = 0; i < NUM_END; i++)
            count_prime(i, &counter);
    } else if (type == 3) {
#pragma omp parallel for num_threads(NUM_THREADS) schedule(static, 10)
        for (int i = 0; i < NUM_END; i++)
            count_prime(i, &counter);
    } else if (type == 4) {
#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic, 10)
		for (int i = 0; i < NUM_END; i++)
            count_prime(i, &counter);
    }

	t2 = omp_get_wtime();
    
	printf("Program Execution Time : %lfms\n", (t2 - t1) * 1000);
	printf("1...%d prime# counter=%d\n", NUM_END - 1, counter);
	return 0;
}