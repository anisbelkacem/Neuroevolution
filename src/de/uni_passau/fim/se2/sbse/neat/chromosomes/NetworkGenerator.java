package de.uni_passau.fim.se2.sbse.neat.chromosomes;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.InnovationImpl;

import java.util.*;

import static java.util.Objects.requireNonNull;

/**
 * Creates fully connected feed-forward neural networks consisting of one input and one output layer.
 */
public class NetworkGenerator {

    /**
     * The number of desired input neurons.
     */
    private final int inputSize;

    /**
     * The number of desired output neurons.
     */
    private final int outputSize;

    /**
     * The random number generator.
     */
    private final Random random;

    /**
     * The set of innovations that occurred so far in the search.
     * Novel innovations created during the generation of the network must be added to this set.
     */
    private final Set<Innovation> innovations;

    /**
     * Creates a new network generator.
     *
     * @param innovations The set of innovations that occurred so far in the search.
     * @param inputSize   The number of desired input neurons.
     * @param outputSize  The number of desired output neurons.
     * @param random      The random number generator.
     * @throws NullPointerException if the random number generator is {@code null}.
     */
    public NetworkGenerator(Set<Innovation> innovations, int inputSize, int outputSize, Random random) {
        this.innovations = requireNonNull(innovations);
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.random = requireNonNull(random);
    }

    /**
     * Generates a new fully connected feed-forward network chromosome.
     *
     * @return a new network chromosome.
     */
    public NetworkChromosome generate() {
        int i = 0, j = 0;
        Map<Double, List<NeuronGene>> layersMap = new HashMap<>();
        List<ConnectionGene> ListOfconnections = new ArrayList<>();
        List<NeuronGene> in_Neurons = new ArrayList<>();
        List<NeuronGene> out_Neurons = new ArrayList<>();
        int biasId = -1;
        NeuronGene biasNeuron = new NeuronGene(biasId, ActivationFunction.NONE, NeuronType.BIAS);
        in_Neurons.add(biasNeuron);
    

        while (i < inputSize) {
            in_Neurons.add(new NeuronGene(i, ActivationFunction.NONE, NeuronType.INPUT));
            i++;
        }

        while (j < outputSize) {
            out_Neurons.add(new NeuronGene(inputSize + j, ActivationFunction.TANH, NeuronType.OUTPUT));
            j++;
        }
    
        layersMap.put(0.0, in_Neurons);  
        layersMap.put(1.0, out_Neurons); 
        for (NeuronGene outputNeuron : out_Neurons) {
            int innovationNum = InnovationImpl.getInnovationNumber(biasNeuron, outputNeuron); 
            double weight = random.nextDouble() * 2 - 1; 
            ListOfconnections.add(new ConnectionGene(biasNeuron, outputNeuron, weight, true, innovationNum));
        }
        for (NeuronGene inputNeuron : in_Neurons) {
            if (inputNeuron.getNeuronType() == NeuronType.BIAS) continue; 
            
            for (NeuronGene outputNeuron : out_Neurons) {
                int innovationNum = InnovationImpl.getInnovationNumber(inputNeuron, outputNeuron); 
                double weight = random.nextDouble() * 2 - 1;
                ListOfconnections.add(new ConnectionGene(inputNeuron, outputNeuron, weight, true, innovationNum));
            }
        }
        return new NetworkChromosome(layersMap, ListOfconnections);
    }

    public NetworkChromosome generate(String taskType) {
        int i = 0, j = 0;
        Map<Double, List<NeuronGene>> layersMap = new HashMap<>();
        List<ConnectionGene> ListOfconnections = new ArrayList<>();
        List<NeuronGene> in_Neurons = new ArrayList<>();
        List<NeuronGene> out_Neurons = new ArrayList<>();
        int biasId = -1;
        NeuronGene biasNeuron = new NeuronGene(biasId, ActivationFunction.NONE, NeuronType.BIAS);
        in_Neurons.add(biasNeuron);
    
        while (i < inputSize) {
            in_Neurons.add(new NeuronGene(i, ActivationFunction.NONE, NeuronType.INPUT));
            i++;
        }
    
        while (j < outputSize) {
            out_Neurons.add(new NeuronGene(inputSize + j, ActivationFunction.TANH, NeuronType.OUTPUT));
            j++;
        }
    
        layersMap.put(0.0, in_Neurons);  
        layersMap.put(1.0, out_Neurons); 
        
        /*for (NeuronGene outputNeuron : out_Neurons) {
            int innovationNum = InnovationImpl.getInnovationNumber(biasNeuron, outputNeuron);
            double weight = random.nextDouble() * 2 - 1; 
            ListOfconnections.add(new ConnectionGene(biasNeuron, outputNeuron, weight, true, innovationNum));
        }*/
        
        // **Modify network initialization based on task**
        if (taskType.equals("XOR")) {
            //System.out.println("Initializing XOR Task...");
            addSimpleConnections(in_Neurons, out_Neurons, ListOfconnections);
        } else if (taskType.equals("CART")) {
            //System.out.println("Initializing CartPole Task...");
            addMoreHiddenLayers(layersMap, in_Neurons, out_Neurons, ListOfconnections, 2);
        } else if (taskType.equals("CART_RANDOMIZED")) {
            //System.out.println("Initializing Randomized CartPole...");
            addMoreHiddenLayers(layersMap, in_Neurons, out_Neurons, ListOfconnections, random.nextInt(2) + 1);
        }
    
        return new NetworkChromosome(layersMap, ListOfconnections);
    }
    private void addSimpleConnections(List<NeuronGene> in_Neurons, List<NeuronGene> out_Neurons, List<ConnectionGene> ListOfconnections) {
        for (NeuronGene inputNeuron : in_Neurons) {
            if (inputNeuron.getNeuronType() == NeuronType.BIAS) continue;
    
            for (NeuronGene outputNeuron : out_Neurons) {
                int innovationNum = InnovationImpl.getInnovationNumber(inputNeuron, outputNeuron);
                double weight = random.nextDouble() * 2 - 1; // Small random weights
                ListOfconnections.add(new ConnectionGene(inputNeuron, outputNeuron, weight, true, innovationNum));
            }
        }
    }
    
    private void addMoreHiddenLayers(Map<Double, List<NeuronGene>> layersMap, List<NeuronGene> in_Neurons, List<NeuronGene> out_Neurons, List<ConnectionGene> ListOfconnections, int numHiddenLayers) {
        //System.out.println("Adding " + numHiddenLayers + " Hidden Layers...");
    
        List<NeuronGene> previousLayer = in_Neurons;
        for (int h = 1; h <= numHiddenLayers; h++) {
            List<NeuronGene> hiddenNeurons = new ArrayList<>();
            for (int k = 0; k < inputSize; k++) {
                NeuronGene hiddenNeuron = new NeuronGene(100 + h * 10 + k, ActivationFunction.NONE, NeuronType.HIDDEN);
                hiddenNeurons.add(hiddenNeuron);
            }
            layersMap.put(h * 1.0, hiddenNeurons);
            connectLayers(previousLayer, hiddenNeurons, ListOfconnections);
            previousLayer = hiddenNeurons;
        }
    
        connectLayers(previousLayer, out_Neurons, ListOfconnections);
    }
    private void connectLayers(List<NeuronGene> fromLayer, List<NeuronGene> toLayer, List<ConnectionGene> ListOfconnections) {
        for (NeuronGene fromNeuron : fromLayer) {
            for (NeuronGene toNeuron : toLayer) {
                int innovationNum = InnovationImpl.getInnovationNumber(fromNeuron, toNeuron);
                double weight = random.nextDouble() * 2 - 1;
                ListOfconnections.add(new ConnectionGene(fromNeuron, toNeuron, weight, true, innovationNum));
            }
        }
    }
    
    
    
    

    
}
