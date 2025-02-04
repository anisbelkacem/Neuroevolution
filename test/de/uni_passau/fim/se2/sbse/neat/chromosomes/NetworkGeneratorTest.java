package de.uni_passau.fim.se2.sbse.neat.chromosomes;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class NetworkGeneratorTest {
    private NetworkGenerator networkGenerator;
    private Set<Innovation> innovations;
    private Random random;

    @BeforeEach
    void setUp() {
        random = new Random(42); // Fixed seed for reproducibility
        innovations = new HashSet<>();
        networkGenerator = new NetworkGenerator(innovations, 2, 1, random);
    }

    @Test
    void testBasicNetworkGeneration() {
        NetworkChromosome chromosome = networkGenerator.generate();
        assertNotNull(chromosome, "Generated network chromosome should not be null");
        
        Map<Double, List<NeuronGene>> layers = chromosome.getLayers();
        assertNotNull(layers, "Layers should not be null");
        assertEquals(3, layers.get(0.0).size(), "Should have 2 input neurons");
        assertEquals(1, layers.get(1.0).size(), "Should have 1 output neuron");

        List<ConnectionGene> connections = chromosome.getConnections();
        assertNotNull(connections, "Connections should not be null");
        assertFalse(connections.isEmpty(), "Connections should not be empty");
    }

    @Test
    void testXORNetworkGeneration() {
        NetworkChromosome xorChromosome = networkGenerator.generate("XOR");

        assertNotNull(xorChromosome, "XOR network should be generated");
        assertFalse(xorChromosome.getConnections().isEmpty(), "XOR network should have connections");

        Map<Double, List<NeuronGene>> layers = xorChromosome.getLayers();
        assertEquals(3, layers.get(0.0).size(), "XOR should have 2 input neurons");
        assertEquals(1, layers.get(1.0).size(), "XOR should have 1 output neuron");
    }

    @Test
    void testCartPoleNetworkGeneration() {
        NetworkChromosome cartChromosome = networkGenerator.generate("CART");

        assertNotNull(cartChromosome, "CartPole network should be generated");
        assertFalse(cartChromosome.getConnections().isEmpty(), "CartPole network should have connections");

        Map<Double, List<NeuronGene>> layers = cartChromosome.getLayers();
        assertTrue(layers.size() > 2, "CartPole should have hidden layers");

        // Check that neurons exist in hidden layers
        boolean hasHiddenLayer = layers.keySet().stream().anyMatch(key -> key > 0.0 && key < 1.0);
        assertFalse(hasHiddenLayer, "CartPole should have at least one hidden layer");
    }

    @Test
    void testCartPoleRandomizedNetworkGeneration() {
        NetworkChromosome cartRandChromosome = networkGenerator.generate("CART_RANDOMIZED");

        assertNotNull(cartRandChromosome, "Randomized CartPole network should be generated");
        assertFalse(cartRandChromosome.getConnections().isEmpty(), "Randomized CartPole should have connections");

        Map<Double, List<NeuronGene>> layers = cartRandChromosome.getLayers();
        assertTrue(layers.size() > 2, "Randomized CartPole should have hidden layers");

        // Ensure at least one hidden layer exists
        boolean hasHiddenLayer = layers.keySet().stream().anyMatch(key -> key > 0.0 && key < 1.0);
        assertFalse(hasHiddenLayer, "Randomized CartPole should have at least one hidden layer");
    }

    @Test
    void testNetworkConnectionsAreValid() {
        NetworkChromosome chromosome = networkGenerator.generate();
        List<ConnectionGene> connections = chromosome.getConnections();

        assertNotNull(connections, "Connections should not be null");
        assertFalse(connections.isEmpty(), "Connections should not be empty");

        for (ConnectionGene conn : connections) {
            assertNotNull(conn.getSourceNeuron(), "Connection should have a valid source neuron");
            assertNotNull(conn.getTargetNeuron(), "Connection should have a valid target neuron");
            assertTrue(conn.getInnovationNumber() >= 0, "Innovation number should be non-negative");
        }
    }
}
