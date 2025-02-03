package de.uni_passau.fim.se2.sbse.neat.chromosomes;

/**
 * Represents a neuron gene that is part of every NEAT chromosome.
 */
public class NeuronGene {
    private final int id;
    private final ActivationFunction activationFunction;
    private final NeuronType neuronType;

    /**
     * Creates a new neuron with the given ID and activation function.
     *
     * @param id                 The ID of the neuron.
     * @param activationFunction The activation function of the neuron (ignored for bias neurons).
     * @param neuronType         The type of the neuron (INPUT, HIDDEN, OUTPUT, or BIAS).
     */
    public NeuronGene(int id, ActivationFunction activationFunction, NeuronType neuronType) {
        this.id = id;
        this.activationFunction = (neuronType == NeuronType.BIAS) ? null : activationFunction;
        this.neuronType = neuronType;
    }

    public int getId() {
        return id;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public NeuronType getNeuronType() {
        return neuronType;
    }

    /**
     * Returns the output of the neuron.
     * - If this is a Bias Node, it always returns 1.0.
     * - Otherwise, it applies the activation function.
     */
    public double getOutput(double inputSum) {
        if (neuronType == NeuronType.BIAS) {
            return 1.0;  // Bias neurons always output 1
        }
        return activationFunction.apply(inputSum);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        NeuronGene that = (NeuronGene) obj;
        return id == that.id && activationFunction == that.activationFunction && neuronType == that.neuronType;
    }

    @Override
    public int hashCode() {
        return java.util.Objects.hash(id, activationFunction, neuronType);
    }

    @Override
    public String toString() {
        return "NeuronGene{" +
                "id=" + id +
                ", activationFunction=" + (activationFunction == null ? "None (Bias Node)" : activationFunction) +
                ", neuronType=" + neuronType +
                '}';
    }
}
