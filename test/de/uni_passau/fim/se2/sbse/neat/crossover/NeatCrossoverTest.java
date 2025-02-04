package de.uni_passau.fim.se2.sbse.neat.crossover;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class NeatCrossoverTest {
    private NeatCrossover neatCrossover;
    private NetworkChromosome parent1;
    private NetworkChromosome parent2;
    private Random random;

    @BeforeEach
    void setUp() {
        random = new Random(42); // Fixed seed for reproducibility
        neatCrossover = new NeatCrossover(random);

        // Setup Parent 1
        Map<Double, List<NeuronGene>> layers1 = new HashMap<>();
        List<NeuronGene> inputNeurons1 = new ArrayList<>();
        List<NeuronGene> outputNeurons1 = new ArrayList<>();
        List<ConnectionGene> connections1 = new ArrayList<>();

        // Input Neurons
        inputNeurons1.add(new NeuronGene(1, ActivationFunction.NONE, NeuronType.INPUT));
        inputNeurons1.add(new NeuronGene(2, ActivationFunction.NONE, NeuronType.INPUT));

        // Output Neuron
        NeuronGene outputNeuron1 = new NeuronGene(3, ActivationFunction.SIGMOID, NeuronType.OUTPUT);
        outputNeurons1.add(outputNeuron1);

        layers1.put(0.0, inputNeurons1);
        layers1.put(1.0, outputNeurons1);

        // Connections
        connections1.add(new ConnectionGene(inputNeurons1.get(0), outputNeuron1, 1.0, true, 1));
        connections1.add(new ConnectionGene(inputNeurons1.get(1), outputNeuron1, -1.0, true, 2));

        parent1 = new NetworkChromosome(layers1, connections1);
        parent1.setFitness(10.0); // Higher fitness for testing

        // Setup Parent 2 (Slightly different structure)
        Map<Double, List<NeuronGene>> layers2 = new HashMap<>();
        List<NeuronGene> inputNeurons2 = new ArrayList<>(inputNeurons1);
        List<NeuronGene> outputNeurons2 = new ArrayList<>(outputNeurons1);
        List<ConnectionGene> connections2 = new ArrayList<>();

        layers2.put(0.0, inputNeurons2);
        layers2.put(1.0, outputNeurons2);

        // Connections (One connection disabled)
        connections2.add(new ConnectionGene(inputNeurons2.get(0), outputNeuron1, 0.5, false, 1)); // Same innovation number, but disabled
        connections2.add(new ConnectionGene(inputNeurons2.get(1), outputNeuron1, -1.5, true, 2));

        parent2 = new NetworkChromosome(layers2, connections2);
        parent2.setFitness(8.0); // Lower fitness
    }

    @Test
    void testCrossoverBasic() {
        NetworkChromosome offspring = neatCrossover.apply(parent1, parent2);

        assertNotNull(offspring, "Offspring should not be null");
        assertFalse(offspring.getConnections().isEmpty(), "Offspring should have connections");
        assertFalse(offspring.getLayers().isEmpty(), "Offspring should have neurons");

        // Ensure all connections have valid neurons
        for (ConnectionGene conn : offspring.getConnections()) {
            assertNotNull(conn.getSourceNeuron(), "Connection should have a valid source neuron");
            assertNotNull(conn.getTargetNeuron(), "Connection should have a valid target neuron");
        }
    }

    @Test
    void testCrossoverInheritance() {
        NetworkChromosome offspring = neatCrossover.apply(parent1, parent2);

        Set<Integer> offspringInnovations = new HashSet<>();
        for (ConnectionGene conn : offspring.getConnections()) {
            offspringInnovations.add(conn.getInnovationNumber());
        }

        // Parent innovation numbers
        Set<Integer> parentInnovations = new HashSet<>(Arrays.asList(1, 2));

        assertTrue(offspringInnovations.containsAll(parentInnovations),
                "Offspring should inherit connections from parents");
    }

    @Test
    void testDisabledGeneInheritance() {
        NetworkChromosome offspring = neatCrossover.apply(parent1, parent2);

        // Check if the disabled gene in one parent remains disabled in offspring with probability
        boolean foundDisabled = false;
        for (ConnectionGene conn : offspring.getConnections()) {
            if (conn.getInnovationNumber() == 1 && !conn.getEnabled()) {
                foundDisabled = true;
                break;
            }
        }
        assertTrue(foundDisabled, "Disabled genes should sometimes be inherited as disabled");
    }

    @Test
    void testOffspringStructure() {
        NetworkChromosome offspring = neatCrossover.apply(parent1, parent2);

        // The offspring should contain all neurons present in the parents
        Set<Integer> offspringNeuronIds = new HashSet<>();
        for (List<NeuronGene> neurons : offspring.getLayers().values()) {
            for (NeuronGene neuron : neurons) {
                offspringNeuronIds.add(neuron.getId());
            }
        }

        Set<Integer> parentNeuronIds = new HashSet<>();
        for (List<NeuronGene> neurons : parent1.getLayers().values()) {
            for (NeuronGene neuron : neurons) {
                parentNeuronIds.add(neuron.getId());
            }
        }
        for (List<NeuronGene> neurons : parent2.getLayers().values()) {
            for (NeuronGene neuron : neurons) {
                parentNeuronIds.add(neuron.getId());
            }
        }

        assertTrue(offspringNeuronIds.containsAll(parentNeuronIds),
                "Offspring should contain all neurons from both parents");
    }
}
