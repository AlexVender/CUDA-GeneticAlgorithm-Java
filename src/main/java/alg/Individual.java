package alg;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Objects;
import java.util.function.Function;

public class Individual
{
    private BitSet bitSet;
    private int[] paramPoses;
    private Function<Individual, Float> fitnessFunc;
    private Float fitness;


    public Individual(Function<Individual, Float> fitnessFunc, int... sizes)
    {
        this.fitnessFunc = fitnessFunc;
        this.paramPoses = new int[sizes.length + 1];
        paramPoses[0] = 0;
        for (int i = 0; i < sizes.length; i++)
        {
            paramPoses[i+1] = paramPoses[i] + sizes[i];
        }
        this.bitSet = new BitSet(paramPoses[sizes.length]);
    }

    public Individual(Individual individual)
    {
        this.fitnessFunc = individual.fitnessFunc;
        this.bitSet = (BitSet) individual.bitSet.clone();
        this.paramPoses = Arrays.copyOf(individual.paramPoses, individual.paramPoses.length);
    }



    public Individual crossover(Individual parent2, int crosspoint)
    {
        Individual child = new Individual(this);
        for (int i = crosspoint; i < getBitsCnt(); i++)
        {
            child.bitSet.set(i, parent2.bitSet.get(i));
        }

        fitness = null;
        return child;
    }

    public void inverseBit(int i)
    {
        bitSet.flip(i);
        fitness = null;
    }

    public int getParam(int i)
    {
        int param = 0;
        int size = paramPoses[i + 1] - paramPoses[i] - 1;
        boolean isNegate = bitSet.get(paramPoses[i] + size); // sign bit
        for (int k = 0; k < size; k++)
        {
            if (bitSet.get(paramPoses[i] + k))
            {
                param |= 1 << k;
            }
        }

        if (isNegate)
        {
            return -param;
        }
        return param;
    }

    public void setParam(int i, int val)
    {
        int size = paramPoses[i + 1] - paramPoses[i] - 1;
        if (val <= (-1 << size) || val >= (1 << size))
        {
            throw new IndexOutOfBoundsException("Value " + val + " is not in [" +
                    ((-1 << size)+1) + ", " + ((1 << size)-1) + "]");
        }

        boolean isNegative = val < 0;
        if (isNegative)
        {
            val = -val;
        }

        for (int k = 0; k < size; k++)
        {
            bitSet.set(
                    paramPoses[i] + k,
                    (val & (1 << k)) != 0
            );
        }
        bitSet.set(paramPoses[i] + size, isNegative); // sign bit
        fitness = null;
    }

    public float getFloatParam(int i, float precision)
    {
        return getParam(i) / precision;
    }

    public void setFloatParam(int i, float val, float precision)
    {
        setParam(i, (int) (val * precision));
    }


    public boolean getBoolParam(int i)
    {
        return bitSet.get(paramPoses[i]);
    }

    public void setBoolParam(int i, boolean val)
    {
        bitSet.set(paramPoses[i], val);
        fitness = null;
    }

    public int getBitsCnt()
    {
        return paramPoses[paramPoses.length - 1];
    }

    public Float getFitness()
    {
        return fitness;
    }

    public void setFitness(float fitness)
    {
        this.fitness = fitness;
    }

    public Float calcFitness()
    {
        fitness = fitnessFunc.apply(this);
        return fitness;
    }


    public int[] getAllParams()
    {
        int[] params = new int[paramPoses.length - 1];
        for (int i = 0; i < params.length; i++)
        {
            params[i] = getParam(i);
        }
        return params;
    }

    public float[] getAllParamsFloat(float precision)
    {
        float[] params = new float[paramPoses.length - 1];
        for (int i = 0; i < params.length; i++)
        {
            params[i] = getFloatParam(i, precision);
        }
        return params;
    }

    @Override
    public boolean equals(final Object o)
    {
        if (this == o)
        {
            return true;
        }
        if (o == null || getClass() != o.getClass())
        {
            return false;
        }
        final Individual that = (Individual) o;
        return bitSet.equals(that.bitSet) &&
                Arrays.equals(paramPoses, that.paramPoses);
    }

    @Override
    public int hashCode()
    {
        int result = Objects.hash(bitSet);
        result = 31 * result + Arrays.hashCode(paramPoses);
        return result;
    }

    @Override
    public String toString()
    {
        StringBuilder result = new StringBuilder();
        result.append('[').append(getFloatParam(0, 10f));
        for (int i = 1; i < paramPoses.length - 1; i++)
        {
            result.append(", ").append(getFloatParam(i, 10f));
        }
        result.append(']');
        return result.toString();
    }
}
