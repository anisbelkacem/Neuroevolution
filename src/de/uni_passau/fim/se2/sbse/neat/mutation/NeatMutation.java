package de.uni_passau.fim.se2.sbse.neat.mutation;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.InnovationImpl;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;

import java.util.Random;
import java.util.Set;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * Implements the mutation operator for the Neat algorithm, which applies four types of mutations based on probabilities:
 * 1. Add a new neuron to the network.
 * 2. Add a new connection to the network.
 * 3. Mutate the weights of the connections in the network.
 * 4. Toggle the enabled status of a connection in the network.
 */
public class NeatMutation implements Mutation<NetworkChromosome> {

    /**
     * The random number generator to use.
     */
    private final Random random;

    /**
     * The list of innovations that occurred so far in the search.
     * Since Neat applies mutations that change the structure of the network,
     * the set of innovations must be updated appropriately.
     */
    private final Set<Innovation> innovations;
    private int neuronCounter = 1000;
    private int connectionCounter = 1000;
    private final Map<String, Integer> globalInnovationMap;

    /**
     * Constructs a new NeatMutation with the given random number generator and the list of innovations that occurred so far in the search.
     *
     * @param innovations The list of innovations that occurred so far in the search.
     * @param random      The random number generator.
     */
    public NeatMutation(Set<Innovation> innovations, Random random) {
        this.innovations = requireNonNull(innovations);
        this.random = requireNonNull(random);
        this.globalInnovationMap = new HashMap<>();
    }


    /**
     * Applies mutation to the given network chromosome.
     * If a structural mutation is applied, no further non-structural mutations are applied.
     * Otherwise, the weights of the connections are mutated and/or the enabled status of a connection is toggled.
     *
     * @param parent The parent chromosome to mutate.
     * @return The mutated parent chromosome.
     */
    @Override
    public NetworkChromosome apply(NetworkChromosome parent) {
        double mutationChance = random.nextDouble();

        if (mutationChance < 0.25) {
            return addNeuron(parent);
        } else if (mutationChance < 0.50) {
            return addConnection(parent);
        } else if (mutationChance < 0.75) {
            return mutateWeights(parent);
        } else {
            return toggleConnection(parent);
        }
    }


    /**
     * Adds a hidden neuron to the given network chromosome by splitting an existing connection.
     * The connection to be split is chosen randomly from the list of connections in the network chromosome.
     * The connection is disabled and two new connections are added to the network chromosome:
     * One connection with a weight of 1.0 from the source neuron of the split connection to the new hidden neuron,
     * and one connection with the weight of the split connection from the new hidden neuron to the target neuron of the split connection.
     * <p>
     * Since this mutation changes the structure of the network,
     * novel innovations for the new connections must be created if the same mutation has not occurred before.
     * If the same innovation has occurred before, the corresponding innovation numbers must be reused.
     *
     * @param parent The network chromosome to which the new neuron and connections will be added.
     * @return The mutated network chromosome.
     */
    public NetworkChromosome addNeuron(NetworkChromosome parent) {
        List<ConnectionGene> existingConnections = new ArrayList<>(parent.getConnections());
        if (existingConnections.isEmpty()) return parent;

        ConnectionGene chosenConnection = existingConnections.get(random.nextInt(existingConnections.size()));
        ConnectionGene disabledConnection = new ConnectionGene(chosenConnection.getSourceNeuron(), chosenConnection.getTargetNeuron(), chosenConnection.getWeight(), false, chosenConnection.getInnovationNumber());

        NeuronGene addedNeuron = new NeuronGene(neuronCounter++, ActivationFunction.SIGMOID, NeuronType.HIDDEN);

        ConnectionGene inputToNewNeuron = new ConnectionGene(chosenConnection.getSourceNeuron(), addedNeuron, 1.0, true, connectionCounter++);
        ConnectionGene newNeuronToOutput = new ConnectionGene(addedNeuron, chosenConnection.getTargetNeuron(), chosenConnection.getWeight(), true, connectionCounter++);

        innovations.add(new InnovationImpl(connectionCounter - 2));
        innovations.add(new InnovationImpl(connectionCounter - 1));

        List<ConnectionGene> updatedConnections = new ArrayList<>(parent.getConnections());
        updatedConnections.remove(chosenConnection);
        updatedConnections.add(disabledConnection);
        updatedConnections.add(inputToNewNeuron);
        updatedConnections.add(newNeuronToOutput);

        Map<Double, List<NeuronGene>> updatedLayers = new HashMap<>(parent.getLayers());
        updatedLayers.computeIfAbsent(0.5, k -> new ArrayList<>()).add(addedNeuron);

        return new NetworkChromosome(updatedLayers, updatedConnections);
    }

    /**
     * Adds a connection to the given network chromosome.
     * The source neuron of the connection is chosen randomly from the list of neurons in the network chromosome,
     * excluding output neurons.
     * The target neuron of the connection is chosen randomly from the list of neurons in the network chromosome,
     * excluding input and bias neurons.
     * The connection is added to the network chromosome with a random weight between -1.0 and 1.0.
     * The connection must not be recurrent.
     * <p>
     * Since this mutation changes the structure of the network,
     * novel innovations for the new connection must be created if the same mutation has not occurred before.
     * If the same innovation has occurred before, the corresponding innovation number must be reused.
     *
     * @param parent The network chromosome to which the new connection will be added.
     * @return The mutated network chromosome.
     */
    public NetworkChromosome addConnection(NetworkChromosome parent) {
        List<NeuronGene> neuronList = new ArrayList<>();
        parent.getLayers().values().forEach(neuronList::addAll);

        Set<String> existingConnections = new HashSet<>();
        for (ConnectionGene connection : parent.getConnections()) {
            existingConnections.add(connection.getSourceNeuron().getId() + "->" + connection.getTargetNeuron().getId());
        }
    
        int maxAttempts = 100; 
        while (maxAttempts-- > 0) {
            NeuronGene fromNeuron = neuronList.get(random.nextInt(neuronList.size()));
            NeuronGene toNeuron = neuronList.get(random.nextInt(neuronList.size()));
            if (fromNeuron.equals(toNeuron) || 
                fromNeuron.getNeuronType() == NeuronType.OUTPUT || 
                toNeuron.getNeuronType() == NeuronType.INPUT ||
                !isValidFeedForwardConnection(parent, fromNeuron, toNeuron)) {
                continue;
            }
            String connectionSignature = fromNeuron.getId() + "->" + toNeuron.getId();
            if (existingConnections.contains(connectionSignature)) {
                continue;
            }
            existingConnections.add(connectionSignature); 
            int uniqueInnovation;
            if (globalInnovationMap.containsKey(connectionSignature)) {
                uniqueInnovation = globalInnovationMap.get(connectionSignature);
            } else {
                uniqueInnovation = connectionCounter++;
                globalInnovationMap.put(connectionSignature, uniqueInnovation);
                innovations.add(new InnovationImpl(uniqueInnovation));
            }
            List<ConnectionGene> updatedConnections = new ArrayList<>(parent.getConnections());
            updatedConnections.add(new ConnectionGene(fromNeuron, toNeuron, random.nextGaussian() * 0.5, true, uniqueInnovation));
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
            if (entry.getValue().contains(fromNeuron)) {
                fromLayer = entry.getKey();
            }
            if (entry.getValue().contains(toNeuron)) {
                toLayer = entry.getKey();
            }
        }
        
        return fromLayer < toLayer; 
    }
    
    
    

    /**
     * Mutates the weights of the connections in the given network chromosome.
     * The weight is mutated by adding gaussian noise to every weight in the network chromosome.
     *
     * @param parent The network chromosome to mutate.
     * @return The mutated network chromosome.
     */
    public NetworkChromosome mutateWeights(NetworkChromosome parent) {
        List<ConnectionGene> newCon = new ArrayList<>();

        for (ConnectionGene connection : parent.getConnections()) {
            double newWeight = connection.getWeight() + (random.nextGaussian() * 0.1);
            newCon.add(new ConnectionGene(connection.getSourceNeuron(), connection.getTargetNeuron(), newWeight, connection.getEnabled(), connection.getInnovationNumber()));
        }
        return new NetworkChromosome(parent.getLayers(), newCon);
    }

    /**
     * Toggles the enabled status of a random connection in the given network chromosome.
     *
     * @param parent The network chromosome to mutate.
     * @return The mutated network chromosome.
     */
    public NetworkChromosome toggleConnection(NetworkChromosome parent) {
        List<ConnectionGene> connections = new ArrayList<>(parent.getConnections());
        if (connections.isEmpty()) return parent; 
    
        
        int randomIndex = random.nextInt(connections.size());
        ConnectionGene selected = connections.get(randomIndex);
        boolean newStatus = !selected.getEnabled();
        List<ConnectionGene> modifiedConnections = new ArrayList<>();
        for (ConnectionGene conn : connections) {
            if (conn == selected) {
                modifiedConnections.add(new ConnectionGene(
                        conn.getSourceNeuron(),
                        conn.getTargetNeuron(),
                        conn.getWeight(),
                        newStatus,
                        conn.getInnovationNumber()
                ));
            } else {
                modifiedConnections.add(conn);
            }
        }
    
        return new NetworkChromosome(parent.getLayers(), modifiedConnections);
    }
    


}
