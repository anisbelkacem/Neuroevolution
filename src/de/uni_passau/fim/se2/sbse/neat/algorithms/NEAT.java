package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkChromosome;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkGenerator;
import de.uni_passau.fim.se2.sbse.neat.environments.Environment;
import de.uni_passau.fim.se2.sbse.neat.mutation.NeatMutation;
import de.uni_passau.fim.se2.sbse.neat.crossover.NeatCrossover;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.Agent;
import java.util.*;

public class NEAT implements Neuroevolution {
    private final int populationSize;
    private final int maxGenerations;
    private final Random random;
    private Environment environment;
    private final Set<NetworkChromosome> population;
    private final NeatMutation mutation;
    private final NeatCrossover crossover;
    private int generation;
        private double bestFitness = 0;
    private int stagnationCounter = 0;

    public NEAT(int populationSize, int maxGenerations, Random random) {
        this.populationSize = populationSize;
        this.maxGenerations = maxGenerations;
        this.random = random;
        this.population = new HashSet<>();
        this.mutation = new NeatMutation(new HashSet<>(), random);
        this.crossover = new NeatCrossover(random);
        this.generation = 0;
    }

    @Override
    public Agent solve(Environment environment) {
        this.environment = environment;
        initializePopulation();
        while (!environment.solved(getBestAgent()) && generation < maxGenerations) {
            evaluatePopulation();
            nextGeneration();
            System.out.println("Generation " + generation + " Best Fitness: " + getBestAgent().getFitness());
        }
        return getBestAgent();
    }

    @Override
    public int getGeneration() {
        return generation;
    }

    private void initializePopulation() {
        for (int i = 0; i < populationSize; i++) {
            NetworkChromosome chromosome = new NetworkGenerator(new HashSet<>(), environment.stateSize(), environment.actionInputSize(), random).generate();
            population.add(chromosome);
        }
    }

    private void evaluatePopulation() {
        Map<NetworkChromosome, Integer> speciesCounts = new HashMap<>();
        for (NetworkChromosome agent : population) {
            double fitness = environment.evaluate(agent);
            agent.setFitness(fitness);
            speciesCounts.put(agent, speciesCounts.getOrDefault(agent, 0) + 1);
        }
    
        for (NetworkChromosome agent : population) {
            agent.setFitness(agent.getFitness() / speciesCounts.get(agent));  
        }
    }
    

    private NetworkChromosome selectParent() {
        double totalFitness = population.stream().mapToDouble(NetworkChromosome::getFitness).sum();
        double r = random.nextDouble() * totalFitness;
        double cumulativeFitness = 0;
    
        for (NetworkChromosome agent : population) {
            cumulativeFitness += agent.getFitness();
            if (cumulativeFitness >= r) {
                return agent;
            }
        }
        return population.iterator().next();
    }
    
    private void nextGeneration() {
        List<NetworkChromosome> newPopulation = new ArrayList<>();
        while (newPopulation.size() < populationSize) {
            NetworkChromosome parent1 = selectParent();
            NetworkChromosome parent2 = selectParent();
            NetworkChromosome offspring = crossover.apply(parent1, parent2);
            offspring = mutation.apply(offspring);
            newPopulation.add(offspring);
        }
        population.clear();
        population.addAll(newPopulation);
        generation++;
    }

    private NetworkChromosome getBestAgent() {
        return population.stream().max(Comparator.comparingDouble(NetworkChromosome::getFitness)).orElse(null);
    }

}
