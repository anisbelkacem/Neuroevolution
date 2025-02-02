package de.uni_passau.fim.se2.sbse.neat.chromosomes;

/**
 * Represents a connection gene that is part of every NEAT chromosome.
 */
public class ConnectionGene {
    private final NeuronGene sourceNeuron;
    private final NeuronGene targetNeuron;
    private double weight;
    private boolean enabled;
    private final int innovationNumber;

    // TODO: It's your job to implement this class.
    //  Please do not change the signature of the given constructor and methods and ensure to implement them.
    //  You can add additional methods, fields, and constructors if needed.

    /**
     * Creates a new connection gene with the given source and target neuron, weight, enabled flag, and innovation number.
     *
     * @param sourceNeuronGene The source neuron of the connection.
     * @param targetNeuronGene The target neuron of the connection.
     * @param weight           The weight of the connection.
     * @param enabled          Whether the connection is enabled.
     * @param innovationNumber The innovation number of the connection serving as identifier.
     */
    public ConnectionGene(NeuronGene sourceNeuronGene, NeuronGene targetNeuronGene, double weight, boolean enabled, int innovationNumber) {
        this.sourceNeuron = sourceNeuronGene;
        this.targetNeuron = targetNeuronGene;
        this.weight = weight;
        this.enabled = enabled;
        this.innovationNumber = innovationNumber;
    }

    public NeuronGene getSourceNeuron() {
        return sourceNeuron;
    }

    public NeuronGene getTargetNeuron() {
        return targetNeuron;
    }

    public double getWeight() {
        return weight;
    }

    public boolean getEnabled() {
        return enabled;
    }

    public int getInnovationNumber() {
        return innovationNumber;
    }

    public void setWeight(double newWeight) {
        this.weight = newWeight;
    }

    public void toggleEnabled() {
        this.enabled = !this.enabled;
    }

    @Override
    public String toString() {
        return "ConnectionGene{" +
                "source=" + sourceNeuron.getId() +
                ", target=" + targetNeuron.getId() +
                ", weight=" + weight +
                ", enabled=" + enabled +
                ", innovationNumber=" + innovationNumber +
                '}';
    }
}
