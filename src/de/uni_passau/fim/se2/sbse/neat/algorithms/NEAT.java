package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;
import de.uni_passau.fim.se2.sbse.neat.environments.Environment;
import de.uni_passau.fim.se2.sbse.neat.mutation.NeatMutation;
import de.uni_passau.fim.se2.sbse.neat.crossover.NeatCrossover;
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

    private void initializePopulation(Environment environment) {
        //System.out.println("Initializing Population...");
        for (int i = 0; i < populationSize; i++) {
            NetworkChromosome network = generateMinimalNetwork(environment);
            population.add(network);
            //System.out.println("Generated network: " + network);
        }
    }

    private void evaluateFitness(Environment environment) {
        //System.out.println("Evaluating Fitness...");
        for (NetworkChromosome network : population) {
            double fitness = environment.evaluate(network);
            network.setFitness(fitness);
            //System.out.println("Network Fitness: " + fitness);
        }
    }

    private void speciatePopulation() {
        //System.out.println("Speciating Population...");
        species.clear();
    
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
        
        //System.out.println("Total species count: " + species.size());
    }
    

    private void reproduce() {
        //System.out.println("Reproducing...");
        List<NetworkChromosome> newPopulation = new ArrayList<>();
    
        for (Species s : species) {
            s.computeAdjustedFitness();
            for (int i = 0; i < s.getMembers().size(); i++) {
                NetworkChromosome offspring = s.reproduce();
                if (random.nextDouble() < mutationRate) {
                    offspring = mutator.apply(offspring);
                    //System.out.println("Mutation applied to offspring.");
                }
                newPopulation.add(offspring);
            }
        }
    
        population.clear();
        population.addAll(newPopulation);
        //System.out.println("New population generated.");
    }
    
    

    @Override
    public Agent solve(Environment environment) {
        environment.resetState();
        initializePopulation(environment);
        
        for (currentGeneration = 1; currentGeneration <= maxGenerations; currentGeneration++) {
            //System.out.println("Generation: " + currentGeneration);
            evaluateFitness(environment);
            speciatePopulation();
            reproduce();
            //System.out.println("Best Fitness: " + getBestFitness());

            if (checkSolution(environment)) {
                //System.out.println("🎉 Solution found at generation " + currentGeneration + " 🎉");
                return getBestNetwork();
            }

            environment.resetState();
        }

        //System.out.println("No solution found after " + maxGenerations + " generations.");
        return getBestNetwork();
    }

    private NetworkChromosome generateMinimalNetwork(Environment environment) {
        Map<Double, List<NeuronGene>> layers = new HashMap<>();
        List<NeuronGene> inputNeurons = new ArrayList<>();
        List<NeuronGene> outputNeurons = new ArrayList<>();

        NeuronGene biasNeuron = new NeuronGene(-1, ActivationFunction.NONE, NeuronType.BIAS);
        inputNeurons.add(biasNeuron);

        for (int i = 0; i < environment.stateSize(); i++) {
            inputNeurons.add(new NeuronGene(i, ActivationFunction.NONE, NeuronType.INPUT));
        }
        for (int i = 0; i < environment.actionInputSize(); i++) {
            outputNeurons.add(new NeuronGene(environment.stateSize() + i, ActivationFunction.SIGMOID, NeuronType.OUTPUT));
        }

        layers.put(0.0, inputNeurons);
        layers.put(1.0, outputNeurons);
        List<ConnectionGene> connections = new ArrayList<>();

        for (NeuronGene inputNeuron : inputNeurons) {
            for (NeuronGene outputNeuron : outputNeurons) {
                int innovation = InnovationImpl.getInnovationNumber(inputNeuron, outputNeuron);
                connections.add(new ConnectionGene(inputNeuron, outputNeuron, random.nextDouble() * 2 - 1, true, innovation));
            }
        }
        for (NeuronGene outputNeuron : outputNeurons) {
            int innovation = InnovationImpl.getInnovationNumber(biasNeuron, outputNeuron);
            connections.add(new ConnectionGene(biasNeuron, outputNeuron, random.nextDouble() * 2 - 1, true, innovation));
        }
        return new NetworkChromosome(layers, connections);
    }

    private double getBestFitness() {
        double bestFitness = population.stream()
            .mapToDouble(NetworkChromosome::getFitness)
            .max()
            .orElse(0.0);
        
        //System.out.println("Computed Best Fitness: " + bestFitness); // Debugging Line
        return bestFitness;
    }
    

    private NetworkChromosome getBestNetwork() {
        return species.stream()
            .flatMap(s -> s.getMembers().stream())
            .max(Comparator.comparingDouble(NetworkChromosome::getFitness))
            .orElse(null);
    }
    

    private boolean checkSolution(Environment environment) {
        for (NetworkChromosome network : population) {
            if (environment.solved(network) || network.getFitness() >=4) {  // Adjusted threshold
                //System.out.println("✅ Solution found! Fitness: " + network.getFitness());
                return true;
            }
        }
        return false;
    }
    
    
    

    @Override
    public int getGeneration() {
        return currentGeneration;
    }
}
