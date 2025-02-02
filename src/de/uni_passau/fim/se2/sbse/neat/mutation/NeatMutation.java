package de.uni_passau.fim.se2.sbse.neat.mutation;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.InnovationImpl;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;

import java.util.Random;
import java.util.Set;

import static java.util.Objects.requireNonNull;

import java.util.ArrayList;
import java.util.List;

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

    /**
     * Constructs a new NeatMutation with the given random number generator and the list of innovations that occurred so far in the search.
     *
     * @param innovations The list of innovations that occurred so far in the search.
     * @param random      The random number generator.
     */
    public NeatMutation(Set<Innovation> innovations, Random random) {
        this.innovations = requireNonNull(innovations);
        this.random = requireNonNull(random);
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
        List<ConnectionGene> connections = new ArrayList<>(parent.getConnections());
        if (connections.isEmpty()) return parent; 
    
        ConnectionGene selected = connections.get(random.nextInt(connections.size()));
    
        ConnectionGene disabledConnection = new ConnectionGene(
            selected.getSourceNeuron(),selected.getTargetNeuron(),selected.getWeight(),
            false,
            selected.getInnovationNumber()
        );
    
        NeuronGene newNeuron = new NeuronGene(
            parent.getLayers().size() + 1,ActivationFunction.SIGMOID,NeuronType.HIDDEN
        );
    
        ConnectionGene inToNeuron = new ConnectionGene(
            selected.getSourceNeuron(),newNeuron,1.0,true,innovations.size()
        );
    
        ConnectionGene neuronToOut = new ConnectionGene(
            newNeuron,selected.getTargetNeuron(),selected.getWeight(),true,
            innovations.size() + 1
        );
    
        innovations.add(new InnovationImpl(innovations.size()));
        innovations.add(new InnovationImpl(innovations.size() + 1));
    
        List<ConnectionGene> newConnections = new ArrayList<>();
        for (ConnectionGene conn : connections) {
            if (conn.equals(selected)) {
                newConnections.add(disabledConnection); 
            } else {
                newConnections.add(conn);
            }
        }
        newConnections.add(inToNeuron);
        newConnections.add(neuronToOut);
    
        return new NetworkChromosome(parent.getLayers(), newConnections);
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
        List<NeuronGene> neurons = new ArrayList<>();
        parent.getLayers().values().forEach(neurons::addAll);
    
        NeuronGene source, target;
        do {
            source = neurons.get(random.nextInt(neurons.size()));
            target = neurons.get(random.nextInt(neurons.size()));
        } while (source.getNeuronType() == NeuronType.OUTPUT || target.getNeuronType() == NeuronType.INPUT || source == target);
        for (ConnectionGene conn : parent.getConnections()) {
            if (conn.getSourceNeuron().equals(source) && conn.getTargetNeuron().equals(target)) {
                return new NetworkChromosome(parent.getLayers(), new ArrayList<>(parent.getConnections())); 
            }
        }
    
        int newInnovationNumber = innovations.size();
        innovations.add(new InnovationImpl(newInnovationNumber));
    
        ConnectionGene newConnection = new ConnectionGene(source, target, random.nextDouble() * 2 - 1, true, newInnovationNumber);
    
        List<ConnectionGene> newConnections = new ArrayList<>(parent.getConnections());
        newConnections.add(newConnection);
    
        return new NetworkChromosome(parent.getLayers(), newConnections);
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
