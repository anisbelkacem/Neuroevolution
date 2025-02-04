package de.uni_passau.fim.se2.sbse.neat.chromosomes;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ConnectionGeneTest {
    private ConnectionGene connectionGene;
    private NeuronGene sourceNeuron;
    private NeuronGene targetNeuron;

    @BeforeEach
    void setUp() {
        sourceNeuron = new NeuronGene(1, ActivationFunction.SIGMOID, NeuronType.INPUT);
        targetNeuron = new NeuronGene(2, ActivationFunction.TANH, NeuronType.OUTPUT);
        connectionGene = new ConnectionGene(sourceNeuron, targetNeuron, 0.5, true, 100);
    }

    @Test
    void testInitialization() {
        assertNotNull(connectionGene, "ConnectionGene should be initialized");
        assertEquals(sourceNeuron, connectionGene.getSourceNeuron(), "Source neuron should match");
        assertEquals(targetNeuron, connectionGene.getTargetNeuron(), "Target neuron should match");
        assertEquals(0.5, connectionGene.getWeight(), "Weight should be initialized correctly");
        assertTrue(connectionGene.getEnabled(), "Connection should be enabled initially");
        assertEquals(100, connectionGene.getInnovationNumber(), "Innovation number should match");
    }

    @Test
    void testSetWeight() {
        connectionGene.setWeight(1.2);
        assertEquals(1.2, connectionGene.getWeight(), "Weight should be updated correctly");
    }

    @Test
    void testToggleEnabled() {
        connectionGene.toggleEnabled();
        assertFalse(connectionGene.getEnabled(), "Connection should be disabled after toggle");

        connectionGene.toggleEnabled();
        assertTrue(connectionGene.getEnabled(), "Connection should be enabled after second toggle");
    }

    @Test
    void testToString() {
        String expected = "ConnectionGene{source=1, target=2, weight=0.5, enabled=true, innovationNumber=100}";
        assertEquals(expected, connectionGene.toString(), "String representation should match expected format");
    }
}
