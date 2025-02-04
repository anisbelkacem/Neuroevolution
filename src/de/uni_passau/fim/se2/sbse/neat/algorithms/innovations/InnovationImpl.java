package de.uni_passau.fim.se2.sbse.neat.algorithms.innovations;

import java.util.HashMap;
import java.util.Map;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.NeuronGene;

/**
 * Concrete implementation of the Innovation interface.
 */
public class InnovationImpl implements Innovation {

    private static int innovationNumber = 1000;// Start from 1 to avoid conflicts with default values
    private static final Map<String, Integer> innovationMap = new HashMap<>();

    /**
     * Computes and retrieves a unique innovation number for a connection between two neurons.
     *
     * @param source The source neuron.
     * @param target The target neuron.
     * @return The unique innovation number for this connection.
     */
    public static synchronized int getInnovationNumber(NeuronGene source, NeuronGene target) {
        String key = source.getId() + "-" + target.getId();
        return innovationMap.computeIfAbsent(key, k -> innovationNumber++);
    }

}
