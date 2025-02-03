package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;
import de.uni_passau.fim.se2.sbse.neat.environments.Environment;
import de.uni_passau.fim.se2.sbse.neat.mutation.NeatMutation;
import de.uni_passau.fim.se2.sbse.neat.crossover.NeatCrossover;

import java.util.*;

public class NEAT implements Neuroevolution {

    private final int populationSize;
    private final int maxGenerations;
    private double compatibilityThreshold;
    private final Random random;
    private final Set<Innovation> innovations;
    private List<NetworkChromosome> population;
    private Map<Integer, List<NetworkChromosome>> species;

    public NEAT(int populationSize, int maxGenerations, double compatibilityThreshold, Random random) {
        this.populationSize = populationSize;
        this.maxGenerations = maxGenerations;
        this.compatibilityThreshold = compatibilityThreshold;
        this.random = random;
        this.innovations = new HashSet<>();
        this.population = new ArrayList<>();
        this.species = new HashMap<>();

    }

    @Override
    public Agent solve(Environment environment) {
        initialisePopulation(environment);

        for (int generation = 0; generation < maxGenerations; generation++) {
            assignToSpecies();
            adjustCompatibilityThreshold();
            evolveSpecies();
            evaluatePopulation(environment);

            NetworkChromosome bestAgent = getBestAgent();
            //System.out.println("Generation " + generation + " Best Fitness: " + bestAgent.getFitness());
            if (environment.solved(bestAgent)) {
                System.out.println("Solution found in generation " + generation);
                return bestAgent;
            }
        }

        return getBestAgent();
    }

    private void initialisePopulation(Environment environment) {
        int inputSize = environment.stateSize();  
        int outputSize = environment.actionInputSize(); 
        
        NetworkGenerator generator = new NetworkGenerator(innovations, inputSize, outputSize, random);
        for (int i = 0; i < populationSize; i++) {
            population.add(generator.generate());
        }
    }
    

    private void assignToSpecies() {
        species.clear();
        for (NetworkChromosome chromosome : population) {
            boolean assigned = false;
            for (int speciesId : species.keySet()) {
                NetworkChromosome representative = species.get(speciesId).get(0);
                if (computeCompatibilityDistance(chromosome, representative) < compatibilityThreshold) {
                    species.get(speciesId).add(chromosome);
                    assigned = true;
                    break;
                }
            }
            if (!assigned) {
                int newSpeciesId = species.size();
                species.put(newSpeciesId, new ArrayList<>(List.of(chromosome)));
            }
        }
    }

    private void evolveSpecies() {
        List<NetworkChromosome> newPopulation = new ArrayList<>();
        NeatMutation mutation = new NeatMutation(innovations, random);
        NeatCrossover crossover = new NeatCrossover(random);

        for (List<NetworkChromosome> members : species.values()) {
            members.sort(Comparator.comparingDouble(NetworkChromosome::getFitness).reversed());
            int eliteSize = Math.max(1, members.size() / 5);
            newPopulation.addAll(members.subList(0, eliteSize));

            while (newPopulation.size() < populationSize) {
                NetworkChromosome parent1 = selectParent(members);
                //NetworkChromosome parent2 = selectParent(members);

                //NetworkChromosome offspring = crossover.apply(parent1, parent2);
                NetworkChromosome offspring = mutation.apply(parent1);
                newPopulation.add(offspring);
            }
        }
        population = newPopulation;
    }
    
    private NetworkChromosome selectParent(List<NetworkChromosome> speciesMembers) {
        double totalFitness = speciesMembers.stream().mapToDouble(NetworkChromosome::getFitness).sum();
        double selectionPoint = random.nextDouble() * totalFitness;
        double runningSum = 0;

        for (NetworkChromosome member : speciesMembers) {
            runningSum += member.getFitness();
            if (runningSum >= selectionPoint) {
                return member;
            }
        }
        return speciesMembers.get(0);
    }

    private void evaluatePopulation(Environment environment) {
        for (NetworkChromosome chromosome : population) {
            chromosome.setFitness(environment.evaluate(chromosome));
        }
    }

    private NetworkChromosome getBestAgent() {
        return population.stream().max(Comparator.comparingDouble(NetworkChromosome::getFitness)).orElseThrow();
    }

    private double computeCompatibilityDistance(NetworkChromosome a, NetworkChromosome b) {
        int disjoint = 0, excess = 0, matching = 0;
        double weightDiff = 0.0;

        Set<Integer> innovationsA = new HashSet<>();
        Set<Integer> innovationsB = new HashSet<>();

        for (ConnectionGene conn : a.getConnections()) {
            innovationsA.add(conn.getInnovationNumber());
        }
        for (ConnectionGene conn : b.getConnections()) {
            innovationsB.add(conn.getInnovationNumber());
        }

        for (int innovation : innovationsA) {
            if (innovationsB.contains(innovation)) {
                matching++;
                weightDiff += Math.abs(a.getConnections().stream()
                        .filter(c -> c.getInnovationNumber() == innovation)
                        .findFirst().get().getWeight() - 
                        b.getConnections().stream()
                        .filter(c -> c.getInnovationNumber() == innovation)
                        .findFirst().get().getWeight());
            } else {
                disjoint++;
            }
        }

        for (int innovation : innovationsB) {
            if (!innovationsA.contains(innovation)) {
                excess++;
            }
        }

        double N = Math.max(innovationsA.size(), innovationsB.size());
        return (disjoint + excess) / N + (weightDiff / Math.max(1, matching));
    }

    private void adjustCompatibilityThreshold() {
        int numSpecies = species.size();
        if (numSpecies < 5) {
            compatibilityThreshold -= 0.3;
        } else if (numSpecies > 10) {
            compatibilityThreshold += 0.3;
        }
        compatibilityThreshold = Math.max(0.5, Math.min(5.0, compatibilityThreshold));  
    }

    @Override
    public int getGeneration() {
        return maxGenerations;
    }
}
