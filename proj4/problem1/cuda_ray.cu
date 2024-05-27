#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define SPHERES 20

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048

struct Sphere
{
    float r, b, g;
    float radius;
    float x, y, z;
    // GPU function called by only Other GPU function
    __device__ float hit(float ox, float oy, float *n)
    {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius)
        {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

// // GPU function called by Other host or global function
__global__ void kernel(Sphere *s, unsigned char *ptr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = x + y * DIM;
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++)
    {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz)
        {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

void ppm_write(unsigned char *bitmap, int xdim, int ydim, FILE *fp)
{
    int i, x, y;
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", xdim, ydim);
    fprintf(fp, "255\n");
    for (y = 0; y < ydim; y++)
    {
        for (x = 0; x < xdim; x++)
        {
            i = x + y * xdim;
            fprintf(fp, "%d %d %d ", bitmap[4 * i], bitmap[4 * i + 1], bitmap[4 * i + 2]);
        }
        fprintf(fp, "\n");
    }
}

int main(int argc, char *argv[])
{
    unsigned char *bitmap;

    srand(time(NULL));

    FILE *fp = fopen("result.ppm", "w");

    clock_t start = clock();

    Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++)
    {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(2000.0f) - 1000;
        temp_s[i].y = rnd(2000.0f) - 1000;
        temp_s[i].z = rnd(2000.0f) - 1000;
        temp_s[i].radius = rnd(200.0f) + 40;
    }

    Sphere *temp_s_dst;
    // allocate gpu memory
    cudaMalloc((void **)&temp_s_dst, sizeof(Sphere) * SPHERES);
    // copy host memory to gpu memory
    cudaMemcpy(temp_s_dst, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);

    bitmap = (unsigned char *)malloc(sizeof(unsigned char) * DIM * DIM * 4);

    unsigned char *bitmap_dst;
    // allocate gpu memory
    cudaMalloc((void **)&bitmap_dst, sizeof(unsigned char) * DIM * DIM * 4);
    // copy host memory to gpu memory
    cudaMemcpy(bitmap_dst, bitmap, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyHostToDevice);

    // set block num and thread num
    dim3 block_num(DIM / 32, DIM / 32);
    dim3 thread_num(32, 32);
    kernel<<<block_num, thread_num>>>(temp_s_dst, bitmap_dst);

    // syncronize kernel excution
    cudaDeviceSynchronize();

    // copy gpu memory to host memory
    cudaMemcpy(temp_s, temp_s_dst, sizeof(Sphere) * SPHERES, cudaMemcpyDeviceToHost);
    cudaMemcpy(bitmap, bitmap_dst, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyDeviceToHost);

    clock_t end = clock();
    ppm_write(bitmap, DIM, DIM, fp);

    fclose(fp);
    free(bitmap);
    free(temp_s);
    cudaFree(bitmap_dst);
    cudaFree(temp_s_dst);

    printf("CUDA ray tracing: %.0f ms\n[result.ppm] was generated.\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    return 0;
}