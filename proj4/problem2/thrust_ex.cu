#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>

using namespace thrust::placeholders;

int N = 1000000000;
double step = 1.0 / (double)N;

struct PiFunction
{
    __host__ __device__ double operator()(const double &x) const
    {
        return 4 / (1 + x * x);
    }
};

int main(){
    clock_t start = clock();

    // declare gpu vector size is N
    thrust::device_vector<double> y(N);
    // fill y begin to y end starting with step * 0.5 jump to step size
    thrust::sequence(y.begin(), y.end(), step * 0.5, step);
    // apply function () operator to each element in y
    thrust::transform(y.begin(), y.end(), y.begin(), PiFunction());
    // multiply step to each element in y, _1 is alias that element itself
    thrust::transform(y.begin(), y.end(), y.begin(), step * _1);
    // sum all elements in gpu vector y
    double pi = thrust::reduce(y.begin(), y.end());

    clock_t end = clock();

    // result
    printf("Execution Time : %fms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    printf("pi=%.10lf\n", pi);
    return 0;
}