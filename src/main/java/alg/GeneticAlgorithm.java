package alg;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.*;
import java.util.function.Function;

import alg.cuda.CudaFitnessCalc;

public class GeneticAlgorithm
{
    public static final int PARAMS_CNT = 5;
    public static final int BITS_PER_PARAM = 9;
    public static final int BITS_CNT = PARAMS_CNT * BITS_PER_PARAM;
    public static final int THREAD_POOL_SIZE = 10;
    public static final int ALG_UI_UPDATE_RATE = 200;

    private static Random random = new Random(System.currentTimeMillis());

    private ExecutorService executorService;

    private AlgCallback callback;
    private CudaFitnessCalc cudaFitnessCalc;

    public GeneticAlgorithm()
    {
        ThreadFactory threadFactory = r ->
        {
            Thread thread = new Thread(r);
            thread.setDaemon(true);
            thread.setPriority(Thread.MAX_PRIORITY);
            return thread;
        };
        executorService = Executors.newWorkStealingPool();
//        executorService = Executors.newFixedThreadPool(THREAD_POOL_SIZE, threadFactory);
    }

    public GeneticAlgorithm(AlgCallback callback)
    {
        this();
        this.callback = callback;
    }

    public float fitness(Individual individual, final float[][] points)
    {
        float fitness = 0f;
        for (int i = 0; i < points[0].length; i++)
        {
            float x = points[0][i];
            float y = points[1][i];

            float fApprox = individual.getParam(PARAMS_CNT-1);
            for (int k = PARAMS_CNT-2; k >= 0; k--)
            {
                fApprox = fApprox*x + individual.getParam(k);
            }
            fApprox /= 10f;

            fitness += Math.abs(fApprox - y);
        }
        return fitness / points[0].length;
    }

    private List<Individual> crossover(List<Individual> population)
    {
        int size = population.size() / 2;

        for (int i = 0; i < size; i++)
        {
            int parent1Id = random.nextInt(size);
            int parent2Id = random.nextInt(size);
            while (parent1Id == parent2Id)
            {
                parent2Id = random.nextInt(size);
            }
            Individual parent1 = population.get(parent1Id);
            Individual parent2 = population.get(parent2Id);
            if (parent1.equals(parent2))
            {
                continue;
            }

            int crosspoint = random(1, BITS_CNT - 1); //random between 1 and individual_size_in_bits -1;
            Individual child1 = parent1.crossover(parent2, crosspoint);
            Individual child2 = parent2.crossover(parent1, crosspoint);
            population.add(child1);
            population.add(child2);
        }
        return population;
    }


    private List<Individual> mutation(List<Individual> population)
    {
        for (int i = 1; i < population.size(); i++)
        {
            final Individual individual = population.get(i);
            final int mutBitsCount = random(1, 5);
            final Set<Integer> bits = new HashSet<>();
            while (bits.size() < mutBitsCount)
            {
                int bitIndex = random(0, BITS_CNT - 1);
                if (bits.add(bitIndex))
                {
                    individual.inverseBit(bitIndex);
                }
            }
        }
        return population;
    }

    private List<Individual> selection(List<Individual> population)
    {
//        final int THREAD_CALC_SIZE = population.size() / THREAD_POOL_SIZE;
//        List<Callable<Object>> tasks = new ArrayList<>(THREAD_POOL_SIZE);
//        for (int k = 0; k < THREAD_POOL_SIZE; k++)
//        {
//            final int fromPos = THREAD_CALC_SIZE * k;
//            tasks.add(Executors.callable(() ->
//            {
//                for (int i = fromPos; i < fromPos + THREAD_CALC_SIZE; i++)
//                {
//                    final Individual individual = population.get(i);
//                    individual.calcFitness();
//                }
//            }));
//        }
//        try
//        {
//            executorService.invokeAll(tasks);
//        }
//        catch (InterruptedException e)
//        {
//            throw new RuntimeException(e);
//        }
        population.sort(Comparator.comparing(Individual::getFitness));
        int halfSize = population.size() / 2;
        for (int i = population.size()-1; i >= halfSize; i--)
        {
            population.remove(i);
        }
        return population;
    }


    public Individual execute(float[][] points, int populationSize, int maxGenerationNumber, float aimedFitness)
            throws InterruptedException
    {
        cudaFitnessCalc = new CudaFitnessCalc(points);
        
        List<Individual> population = initPopulation(populationSize, points);
        int generationNumber = 0;
        float bestFitness;

        long timeStart = System.currentTimeMillis();
        long timeCrossover = 0;
        long timeMutation = 0;
        long timeFitnessCalculation = 0;
        long timeSelection = 0;

        do {
            updateUI(population.get(0), generationNumber, timeStart, true);

            generationNumber++;
            timeCrossover -= System.currentTimeMillis();
            crossover(population);
            timeCrossover += System.currentTimeMillis();

            timeMutation -= System.currentTimeMillis();
            mutation(population);
            timeMutation += System.currentTimeMillis();

            timeFitnessCalculation -= System.currentTimeMillis();
            cudaFitnessCalc.calcFitness(population);
            timeFitnessCalculation += System.currentTimeMillis();

            timeSelection -= System.currentTimeMillis();
            selection(population);
            timeSelection += System.currentTimeMillis();

            bestFitness = population.get(0).getFitness();
        } while (generationNumber < maxGenerationNumber && bestFitness > aimedFitness);

        Individual bestIndividual = population.get(0);

        updateUI(bestIndividual, generationNumber, timeStart, false);
        System.out.println("timeCrossover: " + timeCrossover / 1000f + " sec\n" +
                "timeMutation: " + timeMutation / 1000f + " sec\n" +
                "timeFitnessCalculation: " + timeFitnessCalculation / 1000f + " sec\n" +
                "timeSelection: " + timeSelection / 1000f + " sec");

        cudaFitnessCalc.shutdown();
        return bestIndividual;
    }
    
    private void updateUI(final Individual bestIndividual, int generationNumber, long timeStart, boolean isRunning)
            throws InterruptedException
    {
        if (generationNumber % ALG_UI_UPDATE_RATE == 0 || !isRunning)
        {
            if (Thread.interrupted())
            {
                throw new InterruptedException();
            }
            if (generationNumber == 0)
            {
                bestIndividual.calcFitness();
            }

            System.out.println("Generation: " + generationNumber +
                    " \t" + bestIndividual +
                    " \tFitness: " + bestIndividual.getFitness() +
                    " \tElapsed time: " + (System.currentTimeMillis()- timeStart) / 1000f + " sec");
            if (callback != null)
            {
                float[] params = bestIndividual.getAllParamsFloat(10);
                callback.call(params, isRunning, bestIndividual.getFitness(), generationNumber);
            }
        }
    }

    private List<Individual> initPopulation(int populationSize, float[][] points)
    {
        List<Individual> population = new ArrayList<>();
        Function<Individual, Float> fitnessFunc = (individual) -> fitness(individual, points);
        int[] sizes = new int[PARAMS_CNT];
        for (int i = 0; i < sizes.length; i++)
        {
            sizes[i] = BITS_PER_PARAM;
        }

        for (int i = 0; i < populationSize; i++)
        {
            Individual individual = new Individual(fitnessFunc, sizes);
            for (int j = 0; j < PARAMS_CNT; j++)
            {
                individual.setParam(j, random((-1 << (BITS_PER_PARAM-1) - 1), (1 << (BITS_PER_PARAM-1)) - 1));
//                individual.setParam(j, 0);
            }
            population.add(individual);
        }
        return population;
    }

    /**
     * Random value in [a, b]
     */
    private static int random(int a, int b)
    {
        return a + random.nextInt(b + 1 - a);
    }

    public static void main(String[] args) throws InterruptedException
    {
        GeneticAlgorithm geneticAlgorithm = new GeneticAlgorithm();
        final int maxGenerationNumber = 50000;
        final float aimedFitness = 0.1f;

        final int POINTS_CNT = 50;
        float[][] points = new float[2][POINTS_CNT];
        for (int i = 0; i < POINTS_CNT; i++)
        {
            float x = random(-20, 20);
            float y = random(5, 5);
            points[0][i] = x;
            points[1][i] = y;
        }

        CudaFitnessCalc cudaFitnessCalc = new CudaFitnessCalc(points);
        cudaFitnessCalc.calcFitness(geneticAlgorithm.initPopulation(1, points));
        cudaFitnessCalc.shutdown();
//        Individual result = geneticAlgorithm.execute(points, 50, maxGenerationNumber, aimedFitness);

    }
}