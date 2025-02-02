package de.uni_passau.fim.se2.sbse.neat.crossover;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.ConnectionGene;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkChromosome;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NeuronGene;

import java.util.*;
import java.util.stream.Collectors;

import static java.util.Objects.requireNonNull;

/**
 * A NEAT crossover operation that is used by the NEAT algorithm to combine two parent chromosomes.
 */
public class NeatCrossover implements Crossover<NetworkChromosome> {

    private final Random random;

    /**
     * Creates a new NEAT crossover operator with the given random number generator.
     *
     * @param random The random number generator to use.
     */
    public NeatCrossover(Random random) {
        this.random = requireNonNull(random);
    }

    /**
     * Applies a crossover operation to the given parent chromosomes by combining their genes.
     * Matching genes are inherited randomly from either parent, while disjoint/excess genes are inherited from the fitter parent.
     *
     * @param parent1 The first crossover parent.
     * @param parent2 The second crossover parent.
     * @return A new chromosome resulting from the crossover operation.
     */
    @Override
    public NetworkChromosome apply(NetworkChromosome parent1, NetworkChromosome parent2) {
        requireNonNull(parent1);
        requireNonNull(parent2);

        NetworkChromosome fitterParent = parent1.getFitness() >= parent2.getFitness() ? parent1 : parent2;

        Map<Integer, ConnectionGene> connectionsParent1 = parent1.getConnections().stream()
                .collect(Collectors.toMap(ConnectionGene::getInnovationNumber, connection -> connection));
        Map<Integer, ConnectionGene> connectionsParent2 = parent2.getConnections().stream()
                .collect(Collectors.toMap(ConnectionGene::getInnovationNumber, connection -> connection));

        Set<Integer> allInnovationNumbers = new HashSet<>(connectionsParent1.keySet());
        allInnovationNumbers.addAll(connectionsParent2.keySet());

        List<ConnectionGene> offspringConnections = new ArrayList<>();

        for (int innovationNumber : allInnovationNumbers) {
            if (connectionsParent1.containsKey(innovationNumber) && connectionsParent2.containsKey(innovationNumber)) {
                ConnectionGene inheritedGene = random.nextBoolean() ? connectionsParent1.get(innovationNumber) : connectionsParent2.get(innovationNumber);
                offspringConnections.add(copyConnectionGene(inheritedGene));
            } else {
                ConnectionGene inheritedGene = connectionsParent1.containsKey(innovationNumber) ?
                        connectionsParent1.get(innovationNumber) : connectionsParent2.get(innovationNumber);
                if (fitterParent.getConnections().contains(inheritedGene)) {
                    offspringConnections.add(copyConnectionGene(inheritedGene));
                }
            }
        }

        Map<Integer, NeuronGene> neuronsMap = new HashMap<>();
        for (NeuronGene neuron : parent1.getLayers().values().stream().flatMap(List::stream).collect(Collectors.toList())) {
            neuronsMap.put(neuron.getId(), neuron);
        }
        for (NeuronGene neuron : parent2.getLayers().values().stream().flatMap(List::stream).collect(Collectors.toList())) {
            neuronsMap.putIfAbsent(neuron.getId(), neuron); 
        }

        Map<Double, List<NeuronGene>> offspringLayers = new HashMap<>();
        for (NeuronGene neuron : neuronsMap.values()) {
            offspringLayers.computeIfAbsent(getNeuronLayer(neuron, parent1, parent2), k -> new ArrayList<>()).add(neuron);
        }

        return new NetworkChromosome(offspringLayers, offspringConnections);
    }

    private ConnectionGene copyConnectionGene(ConnectionGene original) {
        return new ConnectionGene(
                original.getSourceNeuron(),
                original.getTargetNeuron(),
                original.getWeight(),
                original.getEnabled(),
                original.getInnovationNumber()
        );
    }
    private double getNeuronLayer(NeuronGene neuron, NetworkChromosome parent1, NetworkChromosome parent2) {
        for (Map.Entry<Double, List<NeuronGene>> entry : parent1.getLayers().entrySet()) {
            if (entry.getValue().contains(neuron)) {
                return entry.getKey();
            }
        }
        for (Map.Entry<Double, List<NeuronGene>> entry : parent2.getLayers().entrySet()) {
            if (entry.getValue().contains(neuron)) {
                return entry.getKey();
            }
        }
        throw new IllegalStateException("Neuron does not belong to either parent chromosome.");
    }
}
