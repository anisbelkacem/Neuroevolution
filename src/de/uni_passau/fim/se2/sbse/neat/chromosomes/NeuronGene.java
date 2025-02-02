package de.uni_passau.fim.se2.sbse.neat.chromosomes;

/**
 * Represents a neuron gene that is part of every NEAT chromosome.
 */
public class NeuronGene {
    private final int id;
    private final ActivationFunction activationFunction;
    private final NeuronType neuronType;

    // TODO: It's your job to implement this class.
    //  Please do not change the signature of the given constructor and methods and ensure to implement them.
    //  You can add additional methods, fields, and constructors if needed.

    /**
     * Creates a new neuron with the given ID and activation function.
     *
     * @param id                 The ID of the neuron.
     * @param activationFunction The activation function of the neuron.
     */
    public NeuronGene(int id, ActivationFunction activationFunction, NeuronType neuronType) {
        this.id = id;
        this.activationFunction = activationFunction;
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
                ", activationFunction=" + activationFunction +
                ", neuronType=" + neuronType +
                '}';
    }
    
}
