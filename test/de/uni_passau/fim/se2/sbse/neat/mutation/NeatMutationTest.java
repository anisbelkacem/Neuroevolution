package de.uni_passau.fim.se2.sbse.neat.mutation;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class NeatMutationTest {

    private NeatMutation neatMutation;
    private NetworkChromosome testChromosome;
    private Random random;
    private Set<Innovation> innovations;

    @BeforeEach
    void setUp() {
        random = new Random(42); // Fixed seed for consistency
        innovations = new HashSet<>();
        neatMutation = new NeatMutation(innovations, random);

        // Create a simple initial chromosome
        Map<Double, List<NeuronGene>> layers = new HashMap<>();
        List<NeuronGene> inputNeurons = new ArrayList<>();
        List<NeuronGene> outputNeurons = new ArrayList<>();
        List<ConnectionGene> connections = new ArrayList<>();

        // Add input neurons
        inputNeurons.add(new NeuronGene(1, ActivationFunction.NONE, NeuronType.INPUT));
        inputNeurons.add(new NeuronGene(2, ActivationFunction.NONE, NeuronType.INPUT));

        // Add output neuron
        NeuronGene outputNeuron = new NeuronGene(3, ActivationFunction.SIGMOID, NeuronType.OUTPUT);
        outputNeurons.add(outputNeuron);

        layers.put(0.0, inputNeurons);
        layers.put(1.0, outputNeurons);

        // Add initial connections
        connections.add(new ConnectionGene(inputNeurons.get(0), outputNeuron, 1.0, true, 1));
        connections.add(new ConnectionGene(inputNeurons.get(1), outputNeuron, -1.0, true, 2));

        testChromosome = new NetworkChromosome(layers, connections);
    }

    @Test
    void testMutateWeights() {
        NetworkChromosome mutatedChromosome = neatMutation.mutateWeights(testChromosome);
        assertNotNull(mutatedChromosome, "Mutated chromosome should not be null");
        assertFalse(mutatedChromosome.getConnections().isEmpty(), "Connections should not be empty");

        // Ensure weights have changed
        boolean weightChanged = false;
        for (int i = 0; i < testChromosome.getConnections().size(); i++) {
            double oldWeight = testChromosome.getConnections().get(i).getWeight();
            double newWeight = mutatedChromosome.getConnections().get(i).getWeight();
            if (oldWeight != newWeight) {
                weightChanged = true;
                break;
            }
        }
        assertTrue(weightChanged, "At least one weight should have changed");
    }

    @Test
    void testToggleConnection() {
        NetworkChromosome toggledChromosome = neatMutation.toggleConnection(testChromosome);
        assertNotNull(toggledChromosome, "Toggled chromosome should not be null");

        // Ensure exactly one connection is toggled
        long toggledConnections = testChromosome.getConnections().stream()
                .filter(conn -> toggledChromosome.getConnections().stream()
                        .anyMatch(mutatedConn -> mutatedConn.getInnovationNumber() == conn.getInnovationNumber()
                                && mutatedConn.getEnabled() != conn.getEnabled()))
                .count();

        assertEquals(1, toggledConnections, "Exactly one connection should be toggled");
    }

    @Test
    void testAddNeuron() {
        NetworkChromosome mutatedChromosome = neatMutation.addNeuron(testChromosome);
        assertNotNull(mutatedChromosome, "Mutated chromosome should not be null");
        assertEquals(testChromosome.getConnections().size() + 2, mutatedChromosome.getConnections().size(),
                "Two new connections should be added when a neuron is inserted");
    }

    @Test
    void testAddConnection() {
        NetworkChromosome mutatedChromosome = neatMutation.addConnection(testChromosome);
        assertNotNull(mutatedChromosome, "Mutated chromosome should not be null");
        assertTrue(mutatedChromosome.getConnections().size() >= testChromosome.getConnections().size(),
                "A new connection should be added");
    }
}
