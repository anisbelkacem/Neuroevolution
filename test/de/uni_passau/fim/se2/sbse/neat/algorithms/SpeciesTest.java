package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;
import de.uni_passau.fim.se2.sbse.neat.mutation.NeatMutation;
import de.uni_passau.fim.se2.sbse.neat.crossover.NeatCrossover;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class SpeciesTest {
    private Species species;
    private NetworkChromosome chromosome1;
    private NetworkChromosome chromosome2;
    private NeatMutation mutation;
    private NeatCrossover crossover;
    private Random random;

    @BeforeEach
    void setUp() {
        random = new Random(42); // Fixed seed for reproducibility
        mutation = new NeatMutation(new HashSet<>(), random);
        crossover = new NeatCrossover(random);

        // Create initial chromosome
        Map<Double, List<NeuronGene>> layers = new HashMap<>();
        List<NeuronGene> inputNeurons = new ArrayList<>();
        List<NeuronGene> outputNeurons = new ArrayList<>();
        List<ConnectionGene> connections = new ArrayList<>();

        // Add input neurons
        NeuronGene input1 = new NeuronGene(1, ActivationFunction.NONE, NeuronType.INPUT);
        NeuronGene input2 = new NeuronGene(2, ActivationFunction.NONE, NeuronType.INPUT);
        inputNeurons.add(input1);
        inputNeurons.add(input2);

        // Add output neuron
        NeuronGene outputNeuron = new NeuronGene(3, ActivationFunction.SIGMOID, NeuronType.OUTPUT);
        outputNeurons.add(outputNeuron);

        layers.put(0.0, inputNeurons);
        layers.put(1.0, outputNeurons);

        // Add connections
        connections.add(new ConnectionGene(input1, outputNeuron, 1.0, true, 1));
        connections.add(new ConnectionGene(input2, outputNeuron, -1.0, true, 2));

        chromosome1 = new NetworkChromosome(layers, connections);
        chromosome1.setFitness(10.0); // Higher fitness

        chromosome2 = new NetworkChromosome(layers, connections);
        chromosome2.setFitness(8.0); // Lower fitness

        species = new Species(chromosome1);
    }
    @Test
    void testSelectParentHandlesZeroFitness() {
        chromosome1.setFitness(0);
        chromosome2.setFitness(0);
        species.addMember(chromosome2);

        NetworkChromosome selectedParent = species.selectParent();
        assertNotNull(selectedParent, "Should still select a parent even with zero fitness");
        assertTrue(species.getMembers().contains(selectedParent), "Parent should be part of the species");
    }


    @Test
    void testInitialization() {
        assertNotNull(species, "Species should be initialized");
        assertEquals(1, species.getMembers().size(), "Species should contain the representative as the first member");
    }

    @Test
    void testAddMember() {
        species.addMember(chromosome2);
        assertEquals(2, species.getMembers().size(), "Species should have two members after addition");
    }

    @Test
    void testIsCompatible() {
        assertTrue(species.isCompatible(chromosome2), "Chromosomes with similar structure should be compatible");
    }

    @Test
    void testAdjustFitness() {
        species.addMember(chromosome2);
        species.adjustFitness();
        assertEquals(10.0, species.getMembers().get(0).getFitness(), "Fitness should remain unchanged after adjustment");
    }

    @Test
    void testReproduce() {
        species.addMember(chromosome2);
        List<NetworkChromosome> newPopulation = new ArrayList<>();
        species.reproduce(mutation, crossover, newPopulation, 10, random);

        assertFalse(newPopulation.isEmpty(), "Reproduction should create new members");
        assertTrue(newPopulation.size() > 0, "New population should have at least one offspring");
    }

    @Test
    void testSelectParent() {
        species.addMember(chromosome2);
        NetworkChromosome selectedParent = species.selectParent();
        assertNotNull(selectedParent, "Selected parent should not be null");
        assertTrue(species.getMembers().contains(selectedParent), "Selected parent should be a member of the species");
    }

    @Test
    void testClear() {
        species.addMember(chromosome2);
        species.clear();
        assertEquals(0, species.getMembers().size(), "Species should be empty after clearing");
    }

    @Test
    void testGetRandomMember() {
        species.addMember(chromosome2);
        NetworkChromosome randomMember = species.getRandomMember();
        assertNotNull(randomMember, "Randomly selected member should not be null");
        assertTrue(species.getMembers().contains(randomMember), "Selected member should be part of the species");
    }
    @Test
    void testReproduceWithOneMember() {
        List<NetworkChromosome> newPopulation = new ArrayList<>();
        species.reproduce(mutation, crossover, newPopulation, 10, random);

        assertFalse(newPopulation.isEmpty(), "Reproduction should still work with one member");
        assertEquals(5, newPopulation.size(), "With one member, reproduction should only copy the best agent");
    }

    @Test
    void testReproduceWithZeroFitness() {
        chromosome1.setFitness(0);
        chromosome2.setFitness(0);
        species.addMember(chromosome2);

        List<NetworkChromosome> newPopulation = new ArrayList<>();
        species.reproduce(mutation, crossover, newPopulation, 10, random);

        assertFalse(newPopulation.isEmpty(), "Reproduction should still produce offspring even with zero fitness");
    }
    

}
