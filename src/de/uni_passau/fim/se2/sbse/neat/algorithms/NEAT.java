package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.ActivationFunction;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.Agent;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.ConnectionGene;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkChromosome;
import de.uni_passau.fim.se2.sbse.neat.environments.Environment;
import de.uni_passau.fim.se2.sbse.neat.mutation.NeatMutation;
import de.uni_passau.fim.se2.sbse.neat.crossover.NeatCrossover;
import de.uni_passau.fim.se2.sbse.neat.algorithms.Species;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NeuronGene;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NeuronType;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.InnovationImpl;

import java.util.*;

public class NEAT implements Neuroevolution {
    private final int populationSize;
    private final double mutationRate;
    private final double crossoverRate;
    private final int maxGenerations;
    private final Random random;
    private final List<NetworkChromosome> population;
    private final List<Species> species;
    private final NeatMutation mutator;
    private final NeatCrossover crossover;
    private int currentGeneration;

    public NEAT(int populationSize, double mutationRate, double crossoverRate, int maxGenerations, Random random) {
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.crossoverRate = crossoverRate;
        this.maxGenerations = maxGenerations;
        this.random = random;
        this.population = new ArrayList<>();
        this.species = new ArrayList<>();
        this.mutator = new NeatMutation(new HashSet<>(), random);
        this.crossover = new NeatCrossover(random);
        this.currentGeneration = 0;
    }

    /**
     * Initializes the population with minimal networks (input -> output only).
     */
    private void initializePopulation(Environment environment) {
        for (int i = 0; i < populationSize; i++) {
            NetworkChromosome network = generateMinimalNetwork(environment);
            population.add(network);
        }
    }

    /**
     * Generates a minimal network (no hidden layers, just input -> output).
     */
    private NetworkChromosome generateMinimalNetwork(Environment environment) {
        Map<Double, List<NeuronGene>> layers = new HashMap<>();
        List<NeuronGene> inputNeurons = new ArrayList<>();
        List<NeuronGene> outputNeurons = new ArrayList<>();
    
        // Create Bias Neuron
        NeuronGene biasNeuron = new NeuronGene(-1, ActivationFunction.NONE, NeuronType.BIAS);
        inputNeurons.add(biasNeuron);
    
        // Create Input Neurons (Dynamically set based on the environment's state size)
        for (int i = 0; i < environment.stateSize(); i++) {
            inputNeurons.add(new NeuronGene(i, ActivationFunction.NONE, NeuronType.INPUT));
        }
    
        // Create Output Neuron(s) (Dynamically set based on the environment's action size)
        for (int i = 0; i < environment.actionInputSize(); i++) {
            outputNeurons.add(new NeuronGene(environment.stateSize() + i, ActivationFunction.SIGMOID, NeuronType.OUTPUT));
        }
    
        layers.put(0.0, inputNeurons);
        layers.put(1.0, outputNeurons);
    
        // Create Initial Fully Connected Feedforward Network
        List<ConnectionGene> connections = new ArrayList<>();
        for (NeuronGene inputNeuron : inputNeurons) {
            for (NeuronGene outputNeuron : outputNeurons) {
                int innovation = InnovationImpl.getInnovationNumber(inputNeuron, outputNeuron);
                connections.add(new ConnectionGene(inputNeuron, outputNeuron, random.nextDouble() * 2 - 1, true, innovation));
            }
        }
        for (ConnectionGene conn : connections) {
            System.out.println("Created connection: " + conn.getSourceNeuron().getId() + " -> " + conn.getTargetNeuron().getId() + " (Weight: " + conn.getWeight() + ")");
        }
        
    
        return new NetworkChromosome(layers, connections);
    }
    
    

    

    @Override
    public Agent solve(Environment environment) {
        initializePopulation(environment); 

        for (currentGeneration = 1; currentGeneration <= maxGenerations; currentGeneration++) {
            evaluateFitness(environment);
            speciatePopulation();
            reproduce();

            System.out.println("Generation " + currentGeneration + " - Best Fitness: " + getBestFitness());

            if (checkSolution(environment)) {
                System.out.println("Solution found at generation " + currentGeneration);
                return getBestNetwork();
            }
        }
        return getBestNetwork(); 
    }

    /**
     * Evaluates the fitness of each network in the population.
     */
    private void evaluateFitness(Environment environment) {
        for (NetworkChromosome network : population) {
            double fitness = environment.evaluate(network);
            network.setFitness(fitness);
        }
    }

    /**
     * Groups the population into species based on genetic similarity.
     */
    private void speciatePopulation() {
        for (NetworkChromosome network : population) {
            boolean added = false;
            for (Species s : species) {
                if (s.belongsToSpecies(network)) {
                    s.addMember(network);
                    added = true;
                    break;
                }
            }
            if (!added) {
                species.add(new Species(network));
            }
        }
    }

    /**
     * Reproduces new offspring using crossover and mutation.
     */
    private void reproduce() {
        List<NetworkChromosome> newPopulation = new ArrayList<>();

        for (Species s : species) {
            s.computeAdjustedFitness();
            List<NetworkChromosome> offspring = new ArrayList<>();
            
            for (NetworkChromosome parent : s.getMembers()) {
                if (random.nextDouble() < crossoverRate) {
                    NetworkChromosome parent2 = s.getMembers().get(random.nextInt(s.getMembers().size()));
                    offspring.add(crossover.apply(parent, parent2));
                } else {
                    offspring.add(new NetworkChromosome(parent.getLayers(), parent.getConnections()));
                }
            }

            for (int i = 0; i < offspring.size(); i++) {
                if (random.nextDouble() < mutationRate) {
                    offspring.set(i, mutator.apply(offspring.get(i)));
                }
            }
            newPopulation.addAll(offspring);
        }

        this.population.clear();
        this.population.addAll(newPopulation);
    }

    /**
     * Returns the best fitness score in the population.
     */
    private double getBestFitness() {
        return population.stream().mapToDouble(NetworkChromosome::getFitness).max().orElse(0);
    }

    /**
     * Returns the best network in the population.
     */
    private NetworkChromosome getBestNetwork() {
        return population.stream().max(Comparator.comparingDouble(NetworkChromosome::getFitness)).orElse(null);
    }

    /**
     * Checks if the best network in the population has solved the environment.
     */
    private boolean checkSolution(Environment environment) {
        return population.stream().anyMatch(environment::solved);
    }

    @Override
    public int getGeneration() {
        return currentGeneration;
    }
}
