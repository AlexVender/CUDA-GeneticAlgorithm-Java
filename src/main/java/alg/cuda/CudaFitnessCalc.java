package alg.cuda;


import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

import alg.Individual;
import jcuda.*;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.*;
import jcuda.utils.KernelLauncher;

import java.util.List;


public class CudaFitnessCalc
{
    public static final String KERNELS_PATH = "./kernels/";
    public static final String FITNESS_FILE_NAME = "GPU_fitness_kernel.cu";

    private KernelLauncher fitnessKernel;
    private KernelLauncher reductionKernel;
    private CUdevice device;
    private CUcontext context;

    private int pointsCnt;
    private float[][] points;
    private Pointer pPointsX;
    private Pointer pPointsY;
    private JCudaReduction jCudaReduction;

    public CudaFitnessCalc(float[][] points)
    {
        this.points = points;
        this.pointsCnt = points[0].length;
        
        initGPU();
    
        int pointsSizeInBytes = pointsCnt * Sizeof.FLOAT;
        pPointsX = new Pointer();
        pPointsY = new Pointer();
        cudaMalloc(pPointsX, pointsSizeInBytes);
        cudaMalloc(pPointsY, pointsSizeInBytes);
        cudaMemset(pPointsX, 0, pointsSizeInBytes);
        cudaMemset(pPointsY, 0, pointsSizeInBytes);
        cudaMemcpy(pPointsX, Pointer.to(points[0]), pointsSizeInBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(pPointsY, Pointer.to(points[1]), pointsSizeInBytes, cudaMemcpyHostToDevice);
    }
    
    public void calcFitness(List<Individual> population)
    {
        int n = population.size();
        
        int[] populationFloat = convertPopulation(population);
        
        int populationsSizeInBytes = n * 5 * Sizeof.INT;
        int outputSizeInBytes = n * Sizeof.FLOAT;
        Pointer pDevicePopulation = new Pointer();
        Pointer pDeviceFitnessOutput = new Pointer();

        cudaMalloc(pDevicePopulation, populationsSizeInBytes);
        cudaMalloc(pDeviceFitnessOutput, outputSizeInBytes);

        cudaMemcpy(pDevicePopulation, Pointer.to(populationFloat), populationsSizeInBytes, cudaMemcpyHostToDevice);

        int THREADS_CNT = Math.min(n, 512);
        int BLOCKS_CNT = (int) Math.ceil((double) n / THREADS_CNT);
//        int BLOCKS_CNT = Math.min(n, 128);

        dim3 GRID_SIZE = new dim3(BLOCKS_CNT, 1, 1);
        dim3 BLOCK_SIZE = new dim3(THREADS_CNT, 1, 1);
        fitnessKernel.setup(GRID_SIZE, BLOCK_SIZE)
                .call(n, pDevicePopulation, pointsCnt, pPointsX, pPointsY, pDeviceFitnessOutput);

        float[] fitnessRaw = new float[n];
        cudaMemcpy(Pointer.to(fitnessRaw), pDeviceFitnessOutput, outputSizeInBytes, cudaMemcpyDeviceToHost);

        int i = 0;
        for (Individual individual : population)
        {
//            float fitness = jCudaReduction.reduce(pDeviceFitnessOutput.withByteOffset(i * pointsCnt * Sizeof.FLOAT), pointsCnt) / pointsCnt;
//            float fitness = jCudaReduction.reduceHostKahan(fitnessRaw, i * pointsCnt, pointsCnt) / pointsCnt;
//            float fitness = jCudaReduction.reduceHostNaive(fitnessRaw, i * pointsCnt, pointsCnt) / pointsCnt;
            float fitness = fitnessRaw[i];
            individual.setFitness(fitness);
            i++;
        }

        cudaFree(pDevicePopulation);
        cudaFree(pDeviceFitnessOutput);
    }

    private int[] convertPopulation(List<Individual> population)
    {
        int[] populationArr = new int[population.size() * 5];
        for (int i = 0; i < population.size(); i++)
        {
            Individual individual = population.get(i);
            int[] params = individual.getAllParams();

            System.arraycopy(params, 0, populationArr, 5 * i, 5);
        }
        return populationArr;
    }
    
    private void initGPU() {
        cuInit(0);
        
        device = new CUdevice();
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
        cuDeviceGet(device, 0);
        JCudaDriver.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);
        
        String args = "-D AVOIDBANKCONFLICTS=" + 0 + " ";
        
//        System.out.println("Loading kernels from " + FITNESS_FILE_NAME);
        fitnessKernel = KernelLauncher.create(
                KERNELS_PATH + FITNESS_FILE_NAME,
                "fitness_kernel", args);

        reductionKernel = KernelLauncher.create(
                KERNELS_PATH + "reduction.cu",
                "reduce", args);

        jCudaReduction = new JCudaReduction(reductionKernel);
    }

    public void shutdown()
    {
        cudaFree(pPointsX);
        cudaFree(pPointsY);
        if (context != null)
        {
//            cuCtxDestroy(context);
        }
    }

    public void finalize()
    {
        shutdown();
    }
}
