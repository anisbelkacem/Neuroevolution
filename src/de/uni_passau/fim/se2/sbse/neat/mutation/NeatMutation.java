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
    private static InnovationImpl inn;

    public NeatMutation(Set<Innovation> innovations, Random random) {
        this.innovations = requireNonNull(innovations);
        this.random = requireNonNull(random);
    }

    @Override
    public NetworkChromosome apply(NetworkChromosome parent) {

        return mutateWeights(parent);
        //return toggleConnection(parent);
        //return addNeuron(parent);
        //return addConnection(parent);
        /*if (mutationChance < 0.25) {
            System.out.println("add weight");

            return mutateWeights(parent);
        } else if (mutationChance < 0.50) {
            System.out.println("add toggle");

            return toggleConnection(parent);
        } else if (mutationChance < 0.75) {
            System.out.println("add connection");

            return addConnection(parent);
        } else {
            System.out.println("add Neuron");
            return addNeuron(parent);
        }*/
    }

    private synchronized int getOrCreateInnovationNumber(NeuronGene from, NeuronGene to) {
        return InnovationImpl.getInnovationNumber(from, to);  
    }

    public NetworkChromosome addNeuron(NetworkChromosome parent) {
        List<ConnectionGene> existingConnections = new ArrayList<>(parent.getConnections());
        if (existingConnections.isEmpty()) return parent;
    
        // Select a random connection, ensuring it's not from/to the bias neuron
        ConnectionGene chosenConnection;
        do {
            chosenConnection = existingConnections.get(random.nextInt(existingConnections.size()));
        } while (chosenConnection.getSourceNeuron().getNeuronType() == NeuronType.BIAS || 
                 chosenConnection.getTargetNeuron().getNeuronType() == NeuronType.BIAS);
    
        // Disable the chosen connection
        ConnectionGene disabledConnection = new ConnectionGene(
            chosenConnection.getSourceNeuron(),
            chosenConnection.getTargetNeuron(),
            chosenConnection.getWeight(),
            false,  // Disabling the connection
            chosenConnection.getInnovationNumber()
        );
    
        NeuronGene addedNeuron = new NeuronGene(neuronCounter++, ActivationFunction.SIGMOID, NeuronType.HIDDEN);
        int inputToNewInnovation = getOrCreateInnovationNumber(chosenConnection.getSourceNeuron(), addedNeuron);
        int newToOutputInnovation = getOrCreateInnovationNumber(addedNeuron, chosenConnection.getTargetNeuron());
        ConnectionGene inputToNewNeuron = new ConnectionGene(
            chosenConnection.getSourceNeuron(), addedNeuron, chosenConnection.getWeight(), true, inputToNewInnovation
        );
        ConnectionGene newNeuronToOutput = new ConnectionGene(
            addedNeuron, chosenConnection.getTargetNeuron(), 1.0, true, newToOutputInnovation
        );
      
        List<ConnectionGene> updatedConnections = new ArrayList<>(parent.getConnections());
        updatedConnections.remove(chosenConnection);
        updatedConnections.add(disabledConnection);
        updatedConnections.add(inputToNewNeuron);
        updatedConnections.add(newNeuronToOutput);
    
        double sourceLayer = -1, targetLayer = -1;
        for (Map.Entry<Double, List<NeuronGene>> entry : parent.getLayers().entrySet()) {
            if (entry.getValue().contains(chosenConnection.getSourceNeuron())) sourceLayer = entry.getKey();
            if (entry.getValue().contains(chosenConnection.getTargetNeuron())) targetLayer = entry.getKey();
        }
        double newLayer = (sourceLayer + targetLayer) / 2; 
        Map<Double, List<NeuronGene>> updatedLayers = new HashMap<>(parent.getLayers());
        updatedLayers.computeIfAbsent(newLayer, k -> new ArrayList<>()).add(addedNeuron);
    
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
                toNeuron.getNeuronType() == NeuronType.BIAS ||  
                existingConnections.contains(fromNeuron.getId() + "->" + toNeuron.getId()) ||
                !isValidFeedForwardConnection(parent, fromNeuron, toNeuron)) {
                continue;
            }
            int innovationNumber = getOrCreateInnovationNumber(fromNeuron, toNeuron);
    

            List<ConnectionGene> updatedConnections = new ArrayList<>(parent.getConnections());
            updatedConnections.add(new ConnectionGene(fromNeuron, toNeuron, random.nextGaussian() * 0.5, true, innovationNumber));
            Map<Double, List<NeuronGene>> clonedLayers = new HashMap<>();
            for (Map.Entry<Double, List<NeuronGene>> entry : parent.getLayers().entrySet()) {
                clonedLayers.put(entry.getKey(), new ArrayList<>(entry.getValue()));
            }
    
            return new NetworkChromosome(clonedLayers, updatedConnections);
        }
        //System.err.println("Warning: addConnection() could not find a valid connection after 100 attempts.");
        return new NetworkChromosome(new HashMap<>(parent.getLayers()), new ArrayList<>(parent.getConnections()));
    }
    

    private boolean isValidFeedForwardConnection(NetworkChromosome parent, NeuronGene fromNeuron, NeuronGene toNeuron) {
        Map<Double, List<NeuronGene>> layers = parent.getLayers();
        double fromLayer = -1, toLayer = -1;
    
        for (Map.Entry<Double, List<NeuronGene>> entry : layers.entrySet()) {
            if (entry.getValue().contains(fromNeuron)) fromLayer = entry.getKey();
            if (entry.getValue().contains(toNeuron)) toLayer = entry.getKey();
        }
        if (fromLayer >= toLayer) return false; 
    
        return !createsCycle(parent, fromNeuron, toNeuron);
    }
    private boolean createsCycle(NetworkChromosome parent, NeuronGene fromNeuron, NeuronGene toNeuron) {
        Set<Integer> visited = new HashSet<>();
        return dfsCheckCycle(parent, toNeuron, visited, fromNeuron.getId());
    }
    private boolean dfsCheckCycle(NetworkChromosome parent, NeuronGene current, Set<Integer> visited, int targetId) {
        if (current.getId() == targetId) return true;
    
        visited.add(current.getId());
    
        for (ConnectionGene conn : parent.getConnections()) {
            if (conn.getSourceNeuron().getId() == current.getId() && conn.getEnabled()) {
                if (!visited.contains(conn.getTargetNeuron().getId())) {
                    if (dfsCheckCycle(parent, conn.getTargetNeuron(), visited, targetId)) return true;
                }
            }
        }
        return false;
    }
    /*public NetworkChromosome mutateWeights(NetworkChromosome parent) {
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
    }*/
    public NetworkChromosome mutateWeights(NetworkChromosome parent) {
        List<ConnectionGene> updatedConnections = new ArrayList<>();
        for (ConnectionGene connection : parent.getConnections()) {
            double weightChange = (random.nextDouble() < 1) 
                ? (random.nextGaussian() * 5.0)  // Larger updates
                : (random.nextGaussian() * 5.0); // Occasionally make big jumps
            double newWeight = connection.getWeight() + weightChange;
            
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
