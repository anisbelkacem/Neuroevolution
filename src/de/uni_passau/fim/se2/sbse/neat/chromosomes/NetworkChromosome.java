package de.uni_passau.fim.se2.sbse.neat.chromosomes;


import java.util.List;
import java.util.Map;
import java.util.Random;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

/**
 * Represents a network chromosome in the NEAT algorithm.
 */
public class NetworkChromosome implements Agent {

    // TODO: It's your job to implement this class.
    //  Please do not change the signature of the given constructor and methods and ensure to implement them.
    //  You can add additional methods, fields, and constructors if needed.

    public static final double INPUT_LAYER = 0;
    public static final double OUTPUT_LAYER = 1;

    /**
     * Maps the layer number to a list of neurons in that layer, with zero representing the input layer and one the output layer.
     * All hidden layers between the input and output layer are represented by values between zero and one.
     * For instance, if a new neuron gets added between the input and output layer, it might get the layer number 0.5.
     */
    private final Map<Double, List<NeuronGene>> layers;

    /**
     * Hosts all connections of the network.
     */
    private final List<ConnectionGene> connections;
    private double fitness;

    /**
     * Creates a new network chromosome with the given layers and connections.
     *
     * @param layers      The layers of the network.
     * @param connections The connections of the network.
     */
    public NetworkChromosome(Map<Double, List<NeuronGene>> layers, List<ConnectionGene> connections) {
        this.layers = requireNonNull(layers);
        this.connections = requireNonNull(connections);
        this.fitness = 0.0;
    }

    public Map<Double, List<NeuronGene>> getLayers() {
        return layers;
    }

    public List<ConnectionGene> getConnections() {
        return connections;
    }

    @Override
    public List<Double> getOutput(List<Double> state) {
        List<NeuronGene> inputNeurons = layers.get(INPUT_LAYER);

        int expectedInputSize = (int) inputNeurons.stream().filter(n -> n.getNeuronType() == NeuronType.INPUT).count();
        if (state.size() != expectedInputSize) {
            throw new IllegalArgumentException("State size does not match input layer size.");
        }
        Map<Integer, Double> neuronValues = new HashMap<>();
        int index = 0;
        for (NeuronGene neuron : inputNeurons) {
            if (neuron.getNeuronType() == NeuronType.INPUT) {
                neuronValues.put(neuron.getId(), state.get(index++));
            } else if (neuron.getNeuronType() == NeuronType.BIAS) {
                neuronValues.put(neuron.getId(), 1.0); 
            }
        }
        List<Double> sortedKeys = new ArrayList<>(layers.keySet());
        Collections.sort(sortedKeys);

        for (double layerKey : sortedKeys) {
            for (NeuronGene neuron : layers.get(layerKey)) {
                if (neuron.getNeuronType() == NeuronType.INPUT || neuron.getNeuronType() == NeuronType.BIAS) continue;

                double sum = 0.0;
                for (ConnectionGene connection : connections) {
                    if (connection.getTargetNeuron().getId() == neuron.getId() && connection.getEnabled()) {
                        int sourceId = connection.getSourceNeuron().getId();
                        double weight = connection.getWeight();
                        sum += neuronValues.getOrDefault(sourceId, 0.0) * weight;
                    }
                }
                double activatedValue = applyActivation(neuron.getActivationFunction(), sum);
                neuronValues.put(neuron.getId(), activatedValue);
            }
        }

        
        List<Double> outputValues = new ArrayList<>();
        for (NeuronGene outputNeuron : layers.get(OUTPUT_LAYER)) {
            outputValues.add(neuronValues.getOrDefault(outputNeuron.getId(), 0.0));
        }

        return outputValues;
    }

    @Override
    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    @Override
    public double getFitness() {
        return fitness;
    }


    private double applyActivation(ActivationFunction function, double value) {
        switch (function) {
            case SIGMOID:
                return 1.0 / (1.0 + Math.exp(-value));
            case TANH:
                return Math.tanh(value);
            case NONE:
            default:
                return value;
        }
    }
}
