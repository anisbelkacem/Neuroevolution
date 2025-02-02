package de.uni_passau.fim.se2.sbse.neat.mutation;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.InnovationImpl;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;

import java.util.*;

import static java.util.Objects.requireNonNull;

/**
 * Implements mutation operators for the NEAT algorithm.
 */
public class NeatMutation implements Mutation<NetworkChromosome> {

    private final Random random;
    private final Set<Innovation> innovations;
    private int neuronCounter = 1000;
    private int connectionCounter = 1000;
    private final Map<String, Integer> globalInnovationMap;

    public NeatMutation(Set<Innovation> innovations, Random random) {
        this.innovations = requireNonNull(innovations);
        this.random = requireNonNull(random);
        this.globalInnovationMap = new HashMap<>();
    }

    @Override
    public NetworkChromosome apply(NetworkChromosome parent) {
        double mutationChance = random.nextDouble();

        if (mutationChance < 0.35) {  
            return addNeuron(parent);
        } else if (mutationChance < 0.65) {  
            return addConnection(parent);
        } else if (mutationChance < 0.85) { 
            return mutateWeights(parent);
        } else {
            return toggleConnection(parent);
        }
    }


    
    private int getOrCreateInnovationNumber(NeuronGene from, NeuronGene to) {
        String key = from.getId() + "->" + to.getId();
    
        if (globalInnovationMap.containsKey(key)) {
            return globalInnovationMap.get(key);
        } else {
            int newInnovation = connectionCounter++;
            while (innovations.contains(new InnovationImpl(newInnovation))) {
                newInnovation = connectionCounter++;
            }
            globalInnovationMap.put(key, newInnovation);
            innovations.add(new InnovationImpl(newInnovation));
            return newInnovation;
        }
    }
    

    
    public NetworkChromosome addNeuron(NetworkChromosome parent) {
        List<ConnectionGene> existingConnections = new ArrayList<>(parent.getConnections());
        if (existingConnections.isEmpty()) return parent;
    
        ConnectionGene chosenConnection = existingConnections.get(random.nextInt(existingConnections.size()));
    
        ConnectionGene disabledConnection = new ConnectionGene(
            chosenConnection.getSourceNeuron(), 
            chosenConnection.getTargetNeuron(), 
            chosenConnection.getWeight(), 
            false, 
            chosenConnection.getInnovationNumber()
        );
        NeuronGene addedNeuron = new NeuronGene(neuronCounter++, ActivationFunction.SIGMOID, NeuronType.HIDDEN);
        int inputToNewInnovation = getOrCreateInnovationNumber(chosenConnection.getSourceNeuron(), addedNeuron);
        int newToOutputInnovation = getOrCreateInnovationNumber(addedNeuron, chosenConnection.getTargetNeuron());
    
        ConnectionGene inputToNewNeuron = new ConnectionGene(
            chosenConnection.getSourceNeuron(), addedNeuron, 1.0, true, inputToNewInnovation
        );
    
        ConnectionGene newNeuronToOutput = new ConnectionGene(
            addedNeuron, chosenConnection.getTargetNeuron(), chosenConnection.getWeight(), true, newToOutputInnovation
        );
    
        List<ConnectionGene> updatedConnections = new ArrayList<>(parent.getConnections());
        updatedConnections.remove(chosenConnection);
        updatedConnections.add(disabledConnection);
        updatedConnections.add(inputToNewNeuron);
        updatedConnections.add(newNeuronToOutput);
    
        Map<Double, List<NeuronGene>> updatedLayers = new HashMap<>(parent.getLayers());
        updatedLayers.computeIfAbsent(0.5, k -> new ArrayList<>()).add(addedNeuron);
    
        return new NetworkChromosome(updatedLayers, updatedConnections);
    }
    

    
    public NetworkChromosome addConnection(NetworkChromosome parent) {
        List<NeuronGene> neurons = new ArrayList<>();
        parent.getLayers().values().forEach(neurons::addAll);
    
        Set<String> existingConnections = new HashSet<>();
        for (ConnectionGene connection : parent.getConnections()) {
            existingConnections.add(connection.getSourceNeuron().getId() + "->" + connection.getTargetNeuron().getId());
        }
    
        int maxAttempts = 100;
        while (maxAttempts-- > 0) {
            NeuronGene fromNeuron = neurons.get(random.nextInt(neurons.size()));
            NeuronGene toNeuron = neurons.get(random.nextInt(neurons.size()));
    
            if (fromNeuron.equals(toNeuron) ||
                fromNeuron.getNeuronType() == NeuronType.OUTPUT || 
                toNeuron.getNeuronType() == NeuronType.INPUT ||
                !isValidFeedForwardConnection(parent, fromNeuron, toNeuron)) {
                continue;
            }
    
            String connectionKey = fromNeuron.getId() + "->" + toNeuron.getId();
            if (existingConnections.contains(connectionKey)) {
                continue;
            }
    
            existingConnections.add(connectionKey);
            int innovationNumber = getOrCreateInnovationNumber(fromNeuron, toNeuron);
    
            List<ConnectionGene> updatedConnections = new ArrayList<>(parent.getConnections());
            updatedConnections.add(new ConnectionGene(fromNeuron, toNeuron, random.nextGaussian() * 0.5, true, innovationNumber));
    
            Map<Double, List<NeuronGene>> clonedLayers = new HashMap<>();
            for (Map.Entry<Double, List<NeuronGene>> entry : parent.getLayers().entrySet()) {
                clonedLayers.put(entry.getKey(), new ArrayList<>(entry.getValue()));
            }
    
            return new NetworkChromosome(clonedLayers, updatedConnections);
        }
        return new NetworkChromosome(new HashMap<>(parent.getLayers()), new ArrayList<>(parent.getConnections()));
    }
    

    private boolean isValidFeedForwardConnection(NetworkChromosome parent, NeuronGene fromNeuron, NeuronGene toNeuron) {
        Map<Double, List<NeuronGene>> layers = parent.getLayers();
        double fromLayer = -1, toLayer = -1;

        for (Map.Entry<Double, List<NeuronGene>> entry : layers.entrySet()) {
            if (entry.getValue().contains(fromNeuron)) fromLayer = entry.getKey();
            if (entry.getValue().contains(toNeuron)) toLayer = entry.getKey();
        }

        return fromLayer < toLayer;
    }

    
    public NetworkChromosome mutateWeights(NetworkChromosome parent) {
        List<ConnectionGene> updatedConnections = new ArrayList<>();
        for (ConnectionGene connection : parent.getConnections()) {
            double newWeight = connection.getWeight() + (random.nextGaussian() * 0.1);
            updatedConnections.add(new ConnectionGene(
                connection.getSourceNeuron(), 
                connection.getTargetNeuron(), 
                newWeight, 
                connection.getEnabled(), 
                connection.getInnovationNumber()
            ));
        }
        return new NetworkChromosome(parent.getLayers(), updatedConnections);
    }


    public NetworkChromosome toggleConnection(NetworkChromosome parent) {
        List<ConnectionGene> connections = new ArrayList<>(parent.getConnections());
        if (connections.isEmpty()) return parent;

        int index = random.nextInt(connections.size());
        ConnectionGene selected = connections.get(index);

        List<ConnectionGene> updatedConnections = new ArrayList<>();
        for (ConnectionGene conn : connections) {
            if (conn == selected) {
                updatedConnections.add(new ConnectionGene(
                    conn.getSourceNeuron(),
                    conn.getTargetNeuron(),
                    conn.getWeight(),
                    !conn.getEnabled(),
                    conn.getInnovationNumber()
                ));
            } else {
                updatedConnections.add(conn);
            }
        }

        return new NetworkChromosome(parent.getLayers(), updatedConnections);
    }
}
