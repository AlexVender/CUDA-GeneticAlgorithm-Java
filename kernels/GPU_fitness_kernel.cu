// Each thread calculates fitness for one individual
// Result: vector of fitness

extern "C"
__global__ void fitness_kernel(int populationCnt, int *population,
    int pointsCnt, float *pointsX, float *pointsY, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < populationCnt)
    {
        int shift = 5*i;
        float fitness = 0.0f;
        for (int p = 0; p < pointsCnt; p++)
        {
            float fApprox = population[shift + 4];
            for (int k = 3; k >= 0; k--)
            {
                fApprox = fApprox * (*pointsX) + population[shift + k];
            }
            fApprox /= 10.0f;

            ++pointsX;
            fitness += pow(fApprox - *(pointsY++), 2);
        }
        result[i] = fitness / pointsCnt;
    }
}
