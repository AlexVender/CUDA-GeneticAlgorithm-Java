// Each thread calculates part of fitness abs(y_i - y)
// Result: matrix populationCnt*pointsCnt (row - individual; col - part of fitness)

extern "C"
__global__ void fitness_kernel(int populationCnt, int *population,
    int pointsCnt, float *pointsX, float *pointsY, float *result)
{
    int gridDimX = gridDim.x;
    int blockDimX = blockDim.x;

    for (int i = blockIdx.x; i < populationCnt; i += gridDimX)
    {
        const int shift = 5*i;
        for (int p = threadIdx.x; p < pointsCnt; p += blockDimX)
        {
            const float x = pointsX[p];
            const float y = pointsY[p];

            float fApprox = population[shift + 4];
            for (int k = 3; k >= 0; k--)
            {
                fApprox = fApprox * x + population[shift + k];
            }
            fApprox /= 10.0f;
            result[i*pointsCnt + p] = abs(fApprox - y);
        }
    }
}
