package de.uni_passau.fim.se2.sbse.neat.chromosomes;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NeuronGeneTest {
    private NeuronGene inputNeuron;
    private NeuronGene hiddenNeuron;
    private NeuronGene outputNeuron;
    private NeuronGene biasNeuron;

    @BeforeEach
    void setUp() {
        inputNeuron = new NeuronGene(1, ActivationFunction.NONE, NeuronType.INPUT);
        hiddenNeuron = new NeuronGene(2, ActivationFunction.SIGMOID, NeuronType.HIDDEN);
        outputNeuron = new NeuronGene(3, ActivationFunction.TANH, NeuronType.OUTPUT);
        biasNeuron = new NeuronGene(4, null, NeuronType.BIAS); // Bias neurons ignore activation functions
    }

    @Test
    void testInitialization() {
        assertNotNull(inputNeuron, "Input neuron should be initialized");
        assertNotNull(hiddenNeuron, "Hidden neuron should be initialized");
        assertNotNull(outputNeuron, "Output neuron should be initialized");
        assertNotNull(biasNeuron, "Bias neuron should be initialized");

        assertEquals(1, inputNeuron.getId(), "Input neuron ID should be 1");
        assertEquals(NeuronType.INPUT, inputNeuron.getNeuronType(), "Neuron type should be INPUT");

        assertEquals(2, hiddenNeuron.getId(), "Hidden neuron ID should be 2");
        assertEquals(ActivationFunction.SIGMOID, hiddenNeuron.getActivationFunction(), "Hidden neuron should have SIGMOID activation");

        assertEquals(3, outputNeuron.getId(), "Output neuron ID should be 3");
        assertEquals(ActivationFunction.TANH, outputNeuron.getActivationFunction(), "Output neuron should have TANH activation");

        assertEquals(NeuronType.BIAS, biasNeuron.getNeuronType(), "Bias neuron should have BIAS type");
        assertNull(biasNeuron.getActivationFunction(), "Bias neuron should not have an activation function");
    }

    @Test
    void testActivationFunctionBehavior() {
        assertEquals(0.5, hiddenNeuron.getActivationFunction().apply(0.0), "SIGMOID activation should be 0.5 at zero input");
        assertEquals(1.0, biasNeuron.getOutput(10.0), "Bias neuron should always return 1.0");

        double tanhOutput = outputNeuron.getActivationFunction().apply(1.0);
        assertTrue(tanhOutput > 0.0 && tanhOutput < 1.0, "TANH activation should be in the range (-1,1)");
    }

    @Test
    void testGetOutput() {
        assertEquals(0.0, inputNeuron.getOutput(0.0), "Input neuron with NONE activation should return the same input");
        assertEquals(1.0, biasNeuron.getOutput(-5.0), "Bias neuron should always return 1.0");

        double sigmoidOutput = hiddenNeuron.getOutput(0.5);
        assertTrue(sigmoidOutput > 0.5, "SIGMOID activation should return a value > 0.5 for positive inputs");
    }

    @Test
    void testEqualityAndHashCode() {
        NeuronGene duplicateInputNeuron = new NeuronGene(1, ActivationFunction.NONE, NeuronType.INPUT);
        assertEquals(inputNeuron, duplicateInputNeuron, "Neurons with same ID and type should be equal");
        assertEquals(inputNeuron.hashCode(), duplicateInputNeuron.hashCode(), "Hash codes should be the same");
    }

    @Test
    void testToString() {
        String expectedBiasString = "NeuronGene{id=4, activationFunction=None (Bias Node), neuronType=BIAS}";
        assertEquals(expectedBiasString, biasNeuron.toString(), "Bias neuron string representation should match expected format");
    }
}
