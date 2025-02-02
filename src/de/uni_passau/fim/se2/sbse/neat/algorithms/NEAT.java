package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;
import de.uni_passau.fim.se2.sbse.neat.crossover.NeatCrossover;
import de.uni_passau.fim.se2.sbse.neat.mutation.NeatMutation;
import de.uni_passau.fim.se2.sbse.neat.environments.Environment;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;

import java.util.*;
import java.util.stream.Collectors;

public class NEAT implements Neuroevolution {

    private final int populationSize;
    private final int maxGenerations;
    private final double mutationRate;
    private final double crossoverRate;
    private final Random random;
    private final Set<Innovation> innovations;
    
    private List<NetworkChromosome> population;
    private int generation;
    private double COMPATIBILITY_THRESHOLD = 3.0;
    private double COMPATIBILITY_MODIFIER = 0.3;
    private int TARGET_SPECIES_COUNT = 5;
    private final List<Species> speciesList = new ArrayList<>();
    private final int inputSize;
    private final int outputSize;

    public NEAT(int populationSize, int maxGenerations, double mutationRate, double crossoverRate, Random random, int inputSize, int outputSize) {
        this.populationSize = populationSize;
        this.maxGenerations = maxGenerations;
        this.mutationRate = mutationRate;
        this.crossoverRate = crossoverRate;
        this.random = random;
        this.innovations = new HashSet<>();
        this.population = new ArrayList<>();
        this.generation = 0;
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        initializePopulation();
    }

    private void initializePopulation() {
        for (int i = 0; i < populationSize; i++) {
            NetworkChromosome chromosome = new NetworkGenerator(innovations, inputSize, outputSize, random).generate();
            population.add(chromosome);
        }
    }

    private void evaluateFitness(Environment environment) {
        for (NetworkChromosome chromosome : population) {
            double fitness = environment.evaluate(chromosome);
            chromosome.setFitness(fitness);
        }
    }

    private void speciate() {
        List<Species> newSpeciesList = new ArrayList<>();

        for (NetworkChromosome chromosome : population) {
            boolean assigned = false;

            for (Species species : speciesList) {
                if (getCompatibilityDistance(chromosome, species.getRepresentative()) < COMPATIBILITY_THRESHOLD) {
                    species.addMember(chromosome);
                    assigned = true;
                    break;
                }
            }
            if (!assigned) {
                newSpeciesList.add(new Species(chromosome));
            }
        }

        speciesList.clear();
        speciesList.addAll(newSpeciesList);
        adjustCompatibilityThreshold();
    }

    private double getCompatibilityDistance(NetworkChromosome c1, NetworkChromosome c2) {
        Map<Integer, ConnectionGene> genes1 = c1.getConnections().stream()
                .collect(Collectors.toMap(ConnectionGene::getInnovationNumber, g -> g));
        Map<Integer, ConnectionGene> genes2 = c2.getConnections().stream()
                .collect(Collectors.toMap(ConnectionGene::getInnovationNumber, g -> g));

        int maxInnovation = Math.max(
                genes1.keySet().stream().max(Integer::compareTo).orElse(0),
                genes2.keySet().stream().max(Integer::compareTo).orElse(0)
        );

        int excess = 0, disjoint = 0;
        double weightDiff = 0;
        int matchingGenes = 0;

        for (int i = 0; i <= maxInnovation; i++) {
            boolean inC1 = genes1.containsKey(i);
            boolean inC2 = genes2.containsKey(i);

            if (inC1 && inC2) {
                weightDiff += Math.abs(genes1.get(i).getWeight() - genes2.get(i).getWeight());
                matchingGenes++;
            } else if (inC1 || inC2) {
                if (i > maxInnovation) {
                    excess++;
                } else {
                    disjoint++;
                }
            }
        }

        double avgWeightDiff = (matchingGenes > 0) ? weightDiff / matchingGenes : 0;
        int N = Math.max(genes1.size(), genes2.size());

        return ((1.0 * disjoint) / N) + ((1.0 * excess) / N) + (0.4 * avgWeightDiff);
    }

    private void adjustCompatibilityThreshold() {
        if (speciesList.size() < TARGET_SPECIES_COUNT) {
            COMPATIBILITY_THRESHOLD -= COMPATIBILITY_MODIFIER;
        } else if (speciesList.size() > TARGET_SPECIES_COUNT) {
            COMPATIBILITY_THRESHOLD += COMPATIBILITY_MODIFIER;
        }
    }

    private void reproduce() {
        List<NetworkChromosome> newPopulation = new ArrayList<>();
        NeatCrossover crossover = new NeatCrossover(random);
        NeatMutation mutation = new NeatMutation(innovations, random);

        while (newPopulation.size() < populationSize) {
            NetworkChromosome parent1 = selectParent();
            NetworkChromosome parent2 = selectParent();

            NetworkChromosome offspring;
            if (random.nextDouble() < crossoverRate) {
                offspring = crossover.apply(parent1, parent2);
            } else {
                offspring = parent1;
            }

            if (random.nextDouble() < mutationRate) {
                offspring = mutation.apply(offspring);
            }

            newPopulation.add(offspring);
        }
        population = newPopulation;
    }

    private NetworkChromosome selectParent() {
        int tournamentSize = 3;
        List<NetworkChromosome> tournament = new ArrayList<>();
        for (int i = 0; i < tournamentSize; i++) {
            tournament.add(population.get(random.nextInt(population.size())));
        }
        return Collections.max(tournament, Comparator.comparingDouble(NetworkChromosome::getFitness));
    }

    @Override
    public Agent solve(Environment environment) {
        for (generation = 0; generation < maxGenerations; generation++) {
            evaluateFitness(environment);
            speciate();
            reproduce();

            if (isSolved(environment)) {
                break;
            }
        }
        return getBestAgent();
    }

    private Agent getBestAgent() {
        return Collections.max(population, Comparator.comparingDouble(NetworkChromosome::getFitness));
    }

    private boolean isSolved(Environment environment) {
        return population.stream().anyMatch(environment::solved);
    }

    @Override
    public int getGeneration() {
        return generation;
    }
}
