package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.Agent;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkChromosome;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkGenerator;
import de.uni_passau.fim.se2.sbse.neat.environments.Environment;
import de.uni_passau.fim.se2.sbse.neat.mutation.NeatMutation;
import de.uni_passau.fim.se2.sbse.neat.crossover.NeatCrossover;

import java.util.*;

public class NEAT implements Neuroevolution {
    private final int populationSize;
    private final int maxGenerations;
    private final Random random;
    private Environment environment;
    private final List<Species> speciesList;
    private final NeatMutation mutation;
    private final NeatCrossover crossover;
    private int generation;

    public static final double COMPATIBILITY_THRESHOLD = 0.3;
    public static final double EXCESS_COEFFICIENT = 0.5;
    public static final double DISJOINT_COEFFICIENT = 0.5;
    public static final double WEIGHT_COEFFICIENT = 0.1;

    public NEAT(int populationSize, int maxGenerations, Random random) {
        this.populationSize = Math.min(50, populationSize);
        this.maxGenerations = maxGenerations;
        this.random = random;
        this.speciesList = new ArrayList<>();
        this.mutation = new NeatMutation(new HashSet<>(), random);
        this.crossover = new NeatCrossover(random);
        this.generation = 0;
    }
    

    @Override
    public Agent solve(Environment environment) {
        this.environment = environment;
        //System.out.println("creating initial pop! ");
        initializePopulation(environment);
        //System.out.println("intialze Pop Done! ");
        while (!environment.solved(getBestAgent()) && generation < maxGenerations) {
            evaluatePopulation();
            speciatePopulation();
            nextGeneration();
            //System.out.println("Generation " + generation + " Best Fitness: " + getBestAgent().getFitness());
            generation++;
        }
        //System.out.println("solved on " + generation + "with  Best Fitness: " + getBestAgent().getFitness());
        return getBestAgent();
    }

    @Override
    public int getGeneration() {
        return generation;
    }

    private void initializePopulation(Environment environment) {
    Set<Innovation> globalInnovations = new HashSet<>(); 
    String typeofenvironement="";
    if (environment instanceof de.uni_passau.fim.se2.sbse.neat.environments.XOR )typeofenvironement ="XOR";
    if (environment instanceof de.uni_passau.fim.se2.sbse.neat.environments.SinglePoleBalancing )typeofenvironement ="CART";

    for (int i = 0; i < populationSize; i++) {
        NetworkChromosome chromosome = new NetworkGenerator(
            globalInnovations, 
            environment.stateSize(), 
            environment.actionInputSize(), 
            random
        ).generate(typeofenvironement);
        
        addToSpecies(chromosome); 
    }
}


private void evaluatePopulation() {
    //System.out.println("Evaluating population...");

    double maxRawFitness = Double.MIN_VALUE;
    NetworkChromosome bestAgent = null;

    for (Species species : speciesList) {
        //System.out.println("  -> Species " + species.hashCode() + " has " + species.getMembers().size() + " members.");

        double totalFitness = 0.0;

        // **Evaluate each agent's fitness**
        for (NetworkChromosome agent : species.getMembers()) {
            double previousFitness = agent.getFitness(); // Store previous fitness
            double fitness = environment.evaluate(agent); // Get new fitness score
            agent.setFitness(fitness); // **Update fitness**
            
            //System.out.println("    -> Agent " + agent.hashCode() + " | Prev Fitness: " + previousFitness + " | New Fitness: " + fitness);
            
            // **Check if fitness update actually worked**
            if (agent.getFitness() != fitness) {
                //System.err.println("🚨 ERROR: Fitness update failed for Agent " + agent.hashCode() + "!");
            }

            totalFitness += fitness;

            if (fitness > maxRawFitness) {
                maxRawFitness = fitness;
                bestAgent = agent;
            }
        }

        // **Apply Fitness Sharing (Less Aggressive)**
        for (NetworkChromosome agent : species.getMembers()) {
            double sharedFitness = agent.getFitness();
            agent.setFitness(sharedFitness);
        }
    }

    // **Find Best Agent AFTER Fitness Sharing**
    NetworkChromosome finalBestAgent = speciesList.stream()
        .flatMap(species -> species.getMembers().stream())
        .max(Comparator.comparingDouble(NetworkChromosome::getFitness))
        .orElse(null);

    // **Print Debug Info**
    //System.out.println("  -> Highest Raw Fitness Before Normalization: " + maxRawFitness);
    if (finalBestAgent != null) {
        //System.out.println("  -> Best Agent Found! Final Fitness: " + finalBestAgent.getFitness());
    } else {
        //System.err.println("ERROR: No best agent found!");
    }

    //System.out.println("Evaluation complete.");
}




    private void speciatePopulation() {
        //System.out.println("speciate population");
        List<NetworkChromosome> allNetworks = new ArrayList<>();
        for (Species species : speciesList) {
            allNetworks.addAll(species.getMembers());
        }

        speciesList.clear();

        for (NetworkChromosome network : allNetworks) {
            addToSpecies(network);
        }
        //System.out.println("speciate population done ");
    }

    private void addToSpecies(NetworkChromosome chromosome) {
        for (Species species : speciesList) {
            if (species.isCompatible(chromosome)) {
                species.addMember(chromosome);
                //System.out.println("Added chromosome to existing species: " + species.hashCode());
                return;
            }
        }
        
        Species newSpecies = new Species(chromosome);
        speciesList.add(newSpecies);
        //System.out.println("Created new species: " + newSpecies.hashCode());
    }
    

    private void nextGeneration() {
        //System.out.println("---- Creating new generation: " + (generation + 1) + " ----");
        
        List<NetworkChromosome> newPopulation = new ArrayList<>();
        
        //System.out.println("Adjusting fitness for species...");
        for (Species species : speciesList) {
            species.adjustFitness();
        }
    
        //System.out.println("Reproducing new individuals...");
        for (Species species : speciesList) {
            int beforeSize = newPopulation.size();
            
            //System.out.println("  -> Processing Species " + species.hashCode() + " | Current Population: " + beforeSize);
            
            if (species.getMembers().isEmpty()) {
                //System.err.println("WARNING: Species " + species.hashCode() + " is EMPTY before reproduction! Skipping...");
                continue; // Avoid calling reproduce() on an empty species
            }else{
                //System.err.println("Done : Species " + species.hashCode() + " is NOT EMPTY ...");

            }
        
            long startTime = System.currentTimeMillis();
            if (speciesList.isEmpty()) {
                //System.err.println("CRITICAL ERROR: No species exist before reproduction!");
            }
            
            try {
                //System.err.println("trying to reproduce");
                species.reproduce(mutation, crossover, newPopulation, populationSize, random);
            } catch (Exception e) {
                //System.err.println("ERROR: Exception in species reproduction for species " + species.hashCode());
                e.printStackTrace();
                continue;
            }
        
            long endTime = System.currentTimeMillis();
            int afterSize = newPopulation.size();
            
            //System.out.println("  -> Species " + species.hashCode() + " contributed " + (afterSize - beforeSize) + " offspring.");
            //System.out.println("  -> Time taken: " + (endTime - startTime) + " ms");
        
            if (afterSize == beforeSize) {
                //System.err.println("WARNING: Species " + species.hashCode() + " did not contribute new offspring!");
            }
        }
        
        
    
        //System.out.println("Removing empty species...");
        speciesList.removeIf(species -> species.getMembers().isEmpty());
    
        if (speciesList.isEmpty()) {
            //System.err.println("ERROR: No species left! This should not happen.");
        }
    
        //System.out.println("Ensuring population size consistency...");
        while (newPopulation.size() < populationSize) {
            int index = random.nextInt(speciesList.size());
            //System.out.println("  -> Filling gap with a member from species " + speciesList.get(index).hashCode());
            newPopulation.add(speciesList.get(index).getRandomMember());
        }
    
        //System.out.println("Clearing old species...");
        for (Species species : speciesList) {
            species.clear();
        }
    
        //System.out.println("Assigning new individuals to species...");
        for (NetworkChromosome agent : newPopulation) {
            addToSpecies(agent);
        }
    
        //generation++;
        //System.out.println("---- Generation " + generation + " completed. ----");
    }
    

    private NetworkChromosome getBestAgent() {
        NetworkChromosome bestAgent = speciesList.stream()
            .flatMap(species -> species.getMembers().stream())
            .max(Comparator.comparingDouble(NetworkChromosome::getFitness))
            .orElse(null);
    
        if (bestAgent != null) {
            //System.out.println("  -> Best Agent Found! Fitness BEFORE NORMALIZATION: " + bestAgent.getFitness());
        } else {
            //System.err.println("ERROR: No best agent found!");
        }
    
        return bestAgent;
    }
    
}
