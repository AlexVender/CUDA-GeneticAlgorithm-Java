package alg.cuda;

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.dim3;
import jcuda.utils.KernelLauncher;

public class JCudaReduction
{

    /**
     * Temporary memory for the device output
     */
    private CUdeviceptr deviceBuffer;
    private KernelLauncher reductionKernel;

    public JCudaReduction(KernelLauncher reductionKernel)
    {
        this.reductionKernel = reductionKernel;

        // Allocate a chunk of temporary memory (must be at least
        // numberOfBlocks * Sizeof.FLOAT)
        deviceBuffer = new CUdeviceptr();
        cudaMalloc(deviceBuffer, 1024 * Sizeof.FLOAT);
    }

    /**
     * Implementation of a Kahan summation reduction in plain Java
     *
     * @return The reduction result
     */
    public float reduceHostKahan(float[] data, int from, int length)
    {
        float sum = data[from];
        float c = 0.0f;
        for (int i = from + 1; i < from + length; i++)
        {
            float y = data[i] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        return sum;
    }

    public float reduceHostNaive(float[] data, int from, int length)
    {
        float sum = data[from];
        for (int i = from + 1; i < from + length; i++)
        {
            sum += data[i];
        }
        return sum;
    }

    /**
     * Release all resources allocated by this class
     */
    public void shutdown()
    {
        cudaFree(deviceBuffer);
    }

    /**
     * Perform a reduction on the given input, with a default number
     * of threads and blocks, and return the result. <br />
     * <br />
     *
     * @param hostInput The input to reduce
     *
     * @return The reduction result
     */
    public float reduce(float[] hostInput)
    {
        return reduce(hostInput, 128, 64);
    }

    /**
     * Perform a reduction on the given input, with the given number
     * of threads and blocks, and return the result. <br />
     * <br />
     *
     * @param hostInput  The input to reduce
     * @param maxThreads The maximum number of threads per block
     * @param maxBlocks  The maximum number of blocks per grid
     *
     * @return The reduction result
     */
    public float reduce(float[] hostInput, int maxThreads, int maxBlocks)
    {
        // Allocate and fill the device memory
        CUdeviceptr deviceInput = new CUdeviceptr();
        cudaMalloc(deviceInput, hostInput.length * Sizeof.FLOAT);
        cudaMemcpy(deviceInput, Pointer.to(hostInput), hostInput.length * Sizeof.FLOAT, cudaMemcpyHostToDevice);

        // Call reduction on the device memory
        float result = reduce(deviceInput, hostInput.length, maxThreads, maxBlocks);

        // Clean up and return the result
        cudaFree(deviceInput);
        return result;
    }


    /**
     * Performs a reduction on the given device memory with the given
     * number of elements.
     *
     * @param deviceInput The device input memory
     * @param numElements The number of elements to reduce
     *
     * @return The reduction result
     */
    public float reduce(Pointer deviceInput, int numElements)
    {
        return reduce(deviceInput, numElements, 128, 64);
    }

    /**
     * Performs a reduction on the given device memory with the given
     * number of elements and the specified limits for threads and
     * blocks.
     *
     * @param deviceInput The device input memory
     * @param numElements The number of elements to reduce
     * @param maxThreads  The maximum number of threads
     * @param maxBlocks   The maximum number of blocks
     *
     * @return The reduction result
     */
    public float reduce(Pointer deviceInput, int numElements,
                        int maxThreads, int maxBlocks)
    {
        // Determine the number of threads and blocks 
        // (as done in the NVIDIA sample)
        int numBlocks = getNumBlocks(numElements, maxBlocks, maxThreads);
        int numThreads = getNumThreads(numElements, maxThreads);

        // Call the main reduction method
        return reduce(deviceInput, numElements, numThreads, numBlocks, maxThreads, maxBlocks);
    }


    /**
     * Performs a reduction on the given device memory.
     *
     * @param n           The number of elements for the reduction
     * @param numThreads  The number of threads
     * @param numBlocks   The number of blocks
     * @param maxThreads  The maximum number of threads
     * @param maxBlocks   The maximum number of blocks
     * @param deviceInput The input memory
     *
     * @return The reduction result
     */
    private float reduce(Pointer deviceInput, int n,
                         int numThreads, int numBlocks,
                         int maxThreads, int maxBlocks)
    {
        // Perform a "tree like" reduction as in the NVIDIA sample
        reduce(n, numThreads, numBlocks, deviceInput, deviceBuffer);
        int s = numBlocks;
        while (s > 1)
        {
            int threads = getNumThreads(s, maxThreads);
            int blocks = getNumBlocks(s, maxBlocks, maxThreads);

            reduce(s, threads, blocks, deviceBuffer, deviceBuffer);
            s = (s + (threads * 2 - 1)) / (threads * 2);
        }

        float[] result = {0.0f};
        cudaMemcpy(Pointer.to(result), deviceBuffer, Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        return result[0];
    }


    /**
     * Perform a reduction of the specified number of elements in the given
     * device input memory, using the given number of threads and blocks,
     * and write the results into the given output memory.
     *
     * @param size         The size (number of elements)
     * @param threads      The number of threads
     * @param blocks       The number of blocks
     * @param deviceInput  The device input memory
     * @param deviceOutput The device output memory. Its size must at least
     *                     be numBlocks*Sizeof.FLOAT
     */
    private void reduce(int size, int threads, int blocks,
                        Pointer deviceInput, Pointer deviceOutput)
    {
        //System.out.println("Reduce "+size+" elements with "+
        //    threads+" threads in "+blocks+" blocks");

        // Compute the shared memory size (as done in 
        // the NIVIDA sample)
        int sharedMemSize = threads * Sizeof.FLOAT;
        if (threads <= 32)
        {
            sharedMemSize *= 2;
        }

        dim3 gridSize = new dim3(blocks, 1, 1);
        dim3 blockSize = new dim3(threads, 1, 1);
        reductionKernel.setup(gridSize, blockSize, sharedMemSize)
                .call(deviceInput,
                        deviceOutput,
                        size);
//        cuCtxSynchronize();
    }


    /**
     * Compute the number of blocks that should be used for the
     * given input size and limits
     *
     * @param n          The input size
     * @param maxBlocks  The maximum number of blocks
     * @param maxThreads The maximum number of threads
     *
     * @return The number of blocks
     */
    private int getNumBlocks(int n, int maxBlocks, int maxThreads)
    {
        int threads = getNumThreads(n, maxThreads);
        int blocks = (n + (threads * 2 - 1)) / (threads * 2);
        blocks = Math.min(maxBlocks, blocks);
        return blocks;
    }

    /**
     * Compute the number of threads that should be used for the
     * given input size and limits
     *
     * @param n          The input size
     * @param maxThreads The maximum number of threads
     *
     * @return The number of threads
     */
    private int getNumThreads(int n, int maxThreads)
    {
        return n < maxThreads * 2 ?
                nextPow2((n + 1) / 2) :
                maxThreads;
    }

    /**
     * Returns the power of 2 that is equal to or greater than x
     *
     * @param x The input
     *
     * @return The next power of 2
     */
    private int nextPow2(int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }
}
