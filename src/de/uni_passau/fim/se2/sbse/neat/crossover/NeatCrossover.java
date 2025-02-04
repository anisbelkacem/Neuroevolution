package de.uni_passau.fim.se2.sbse.neat.crossover;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.ConnectionGene;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkChromosome;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NeuronGene;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NeuronType;

import java.util.*;
import java.util.stream.Collectors;

import static java.util.Objects.requireNonNull;

public class NeatCrossover implements Crossover<NetworkChromosome> {
    private final Random random;

    public NeatCrossover(Random random) {
        this.random = requireNonNull(random);
    }

    @Override
    public NetworkChromosome apply(NetworkChromosome parent1, NetworkChromosome parent2) {
        requireNonNull(parent1);
        requireNonNull(parent2);

        NetworkChromosome fitterParent = parent1.getFitness() >= parent2.getFitness() ? parent1 : parent2;

        Map<Integer, ConnectionGene> connectionsParent1 = new HashMap<>();
        for (ConnectionGene connection : parent1.getConnections()) {
            connectionsParent1.putIfAbsent(connection.getInnovationNumber(), connection);
        }

        Map<Integer, ConnectionGene> connectionsParent2 = new HashMap<>();
        for (ConnectionGene connection : parent2.getConnections()) {
            connectionsParent2.putIfAbsent(connection.getInnovationNumber(), connection);
        }

        Set<Integer> allInnovationNumbers = new HashSet<>(connectionsParent1.keySet());
        allInnovationNumbers.addAll(connectionsParent2.keySet());

        List<ConnectionGene> offspringConnections = new ArrayList<>();


        for (int innovationNumber : allInnovationNumbers) {
            ConnectionGene inheritedGene;

            if (connectionsParent1.containsKey(innovationNumber) && connectionsParent2.containsKey(innovationNumber)) {
                inheritedGene = random.nextBoolean() ? connectionsParent1.get(innovationNumber) : connectionsParent2.get(innovationNumber);
            } else {

                inheritedGene = connectionsParent1.containsKey(innovationNumber) ? 
                                connectionsParent1.get(innovationNumber) : 
                                connectionsParent2.get(innovationNumber);

                if (!fitterParent.getConnections().contains(inheritedGene)) {
                    continue; 
                }
            }

            boolean enabled = inheritedGene.getEnabled();
            if (connectionsParent1.containsKey(innovationNumber) && connectionsParent2.containsKey(innovationNumber)) {
                boolean isDisabledInOneParent = !connectionsParent1.get(innovationNumber).getEnabled() || !connectionsParent2.get(innovationNumber).getEnabled();
                if (isDisabledInOneParent && random.nextDouble() < 0.75) {
                    enabled = false;
                }
            }

            offspringConnections.add(new ConnectionGene(
                inheritedGene.getSourceNeuron(),
                inheritedGene.getTargetNeuron(),
                inheritedGene.getWeight(),
                enabled,  
                inheritedGene.getInnovationNumber()
            ));
        }
        

        Map<Integer, NeuronGene> neuronsMap = new HashMap<>();
        for (NeuronGene neuron : parent1.getLayers().values().stream().flatMap(List::stream).collect(Collectors.toList())) {
            neuronsMap.put(neuron.getId(), copyNeuronGene(neuron));
        }
        for (NeuronGene neuron : parent2.getLayers().values().stream().flatMap(List::stream).collect(Collectors.toList())) {
            neuronsMap.putIfAbsent(neuron.getId(), copyNeuronGene(neuron));
        }

        if (neuronsMap.values().stream().noneMatch(n -> n.getNeuronType() == NeuronType.BIAS)) {
            NeuronGene biasNeuron = new NeuronGene(-1, null, NeuronType.BIAS);
            neuronsMap.put(-1, biasNeuron);
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

    private NeuronGene copyNeuronGene(NeuronGene original) {
        return new NeuronGene(original.getId(), original.getActivationFunction(), original.getNeuronType());
    }

    private double getNeuronLayer(NeuronGene neuron, NetworkChromosome parent1, NetworkChromosome parent2) {
        for (Map.Entry<Double, List<NeuronGene>> entry : parent1.getLayers().entrySet()) {
            if (entry.getValue().stream().anyMatch(n -> n.getId() == neuron.getId())) {
                return entry.getKey();
            }
        }
        for (Map.Entry<Double, List<NeuronGene>> entry : parent2.getLayers().entrySet()) {
            if (entry.getValue().stream().anyMatch(n -> n.getId() == neuron.getId())) {
                return entry.getKey();
            }
        }
        return 0.5; 
    }
}
