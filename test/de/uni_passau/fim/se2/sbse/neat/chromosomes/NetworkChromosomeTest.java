package de.uni_passau.fim.se2.sbse.neat.chromosomes;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class NetworkChromosomeTest {
    private NetworkChromosome testChromosome;

    @BeforeEach
    void setUp() {
        // Initialize layers
        Map<Double, List<NeuronGene>> layers = new HashMap<>();
        List<NeuronGene> inputNeurons = new ArrayList<>();
        List<NeuronGene> outputNeurons = new ArrayList<>();
        List<ConnectionGene> connections = new ArrayList<>();

        // Create input neurons
        NeuronGene input1 = new NeuronGene(1, ActivationFunction.NONE, NeuronType.INPUT);
        NeuronGene input2 = new NeuronGene(2, ActivationFunction.NONE, NeuronType.INPUT);
        inputNeurons.add(input1);
        inputNeurons.add(input2);

        // Create output neuron
        NeuronGene outputNeuron = new NeuronGene(3, ActivationFunction.SIGMOID, NeuronType.OUTPUT);
        outputNeurons.add(outputNeuron);

        layers.put(NetworkChromosome.INPUT_LAYER, inputNeurons);
        layers.put(NetworkChromosome.OUTPUT_LAYER, outputNeurons);

        // Create connections
        connections.add(new ConnectionGene(input1, outputNeuron, 1.0, true, 1)); // Weight 1.0
        connections.add(new ConnectionGene(input2, outputNeuron, -1.0, true, 2)); // Weight -1.0

        // Initialize the test chromosome
        testChromosome = new NetworkChromosome(layers, connections);
    }

    @Test
    void testInitialization() {
        assertNotNull(testChromosome, "NetworkChromosome should be initialized");
        assertEquals(2, testChromosome.getLayers().get(NetworkChromosome.INPUT_LAYER).size(), "Should have 2 input neurons");
        assertEquals(1, testChromosome.getLayers().get(NetworkChromosome.OUTPUT_LAYER).size(), "Should have 1 output neuron");
        assertEquals(2, testChromosome.getConnections().size(), "Should have 2 connections");
    }

    @Test
    void testGetConnections() {
        List<ConnectionGene> connections = testChromosome.getConnections();
        assertEquals(2, connections.size(), "Should have 2 connections");
        assertEquals(1.0, connections.get(0).getWeight(), "First connection weight should be 1.0");
        assertEquals(-1.0, connections.get(1).getWeight(), "Second connection weight should be -1.0");
    }

    @Test
    void testForwardPropagation() {
        // Input values: [1.0, 0.0] (should activate first input only)
        List<Double> inputState = Arrays.asList(1.0, 0.0);
        List<Double> output = testChromosome.getOutput(inputState);

        assertEquals(1, output.size(), "Should produce one output value");
        assertTrue(output.get(0) > 0.5, "Output should be positive due to active first input");
    }

    @Test
    void testFitnessAssignment() {
        testChromosome.setFitness(5.0);
        assertEquals(5.0, testChromosome.getFitness(), "Fitness should be set to 5.0");
    }

    @Test
    void testForwardPropagationWithZeroInput() {
        List<Double> inputState = Arrays.asList(0.0, 0.0);
        List<Double> output = testChromosome.getOutput(inputState);

        assertEquals(1, output.size(), "Should produce one output value");
        assertFalse(output.get(0) < 0.5, "Output should be near zero with zero input");
    }
}
