package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkChromosome;
import de.uni_passau.fim.se2.sbse.neat.mutation.NeatMutation;
import de.uni_passau.fim.se2.sbse.neat.crossover.NeatCrossover;

import java.util.*;

public class Species {
    private final NetworkChromosome representative;
    private final List<NetworkChromosome> members;

    public Species(NetworkChromosome representative) {
        this.representative = representative;
        this.members = new ArrayList<>();
        this.members.add(representative);
    }

    public void addMember(NetworkChromosome chromosome) {
        members.add(chromosome);
    }

    public boolean isCompatible(NetworkChromosome chromosome) {
        return calculateCompatibilityDistance(representative, chromosome) < NEAT.COMPATIBILITY_THRESHOLD;
    }

    public void adjustFitness() {
        for (NetworkChromosome member : members) {
            member.setFitness(member.getFitness());
        }
    }

    public void reproduce(NeatMutation mutation, NeatCrossover crossover, List<NetworkChromosome> newPopulation, int totalPopulation, Random random) {
        //System.out.println("  -> Starting reproduction for Species " + this.hashCode() + " | Members: " + members.size());
    
        if (members.isEmpty()) {
            //System.err.println("  -> ERROR: Species " + this.hashCode() + " has no members! Skipping reproduction.");
            return;
        }
    
        NetworkChromosome bestAgent = members.stream().max(Comparator.comparingDouble(NetworkChromosome::getFitness)).orElseThrow();
        newPopulation.add(bestAgent);
        //System.out.println("  -> Best agent added to new population. Fitness: " + bestAgent.getFitness());
    
        int offspringCount = 0;
        while (newPopulation.size() < totalPopulation / 2) {
            NetworkChromosome parent1 = selectParent();
            NetworkChromosome parent2 = selectParent();
    
            //System.out.println("  -> Selected Parents | P1 Fitness: " + parent1.getFitness() + " | P2 Fitness: " + parent2.getFitness());
            
            //System.out.println("  -> applying crossover");
            
            NetworkChromosome offspring = crossover.apply(parent1, parent2);
            //System.out.println("  ->  crossover done , applying mutation ");
            offspring = mutation.apply(offspring);
            //System.out.println("  -> mutation done ");
    
            newPopulation.add(offspring);
            offspringCount++;
            //System.out.println("  -> Offspring " + offspringCount + " added to new population.");
        }
    
        //System.out.println("  -> Species " + this.hashCode() + " reproduction complete. Created " + offspringCount + " offspring.");
    }
    

    public NetworkChromosome selectParent() {
    
        double totalFitness = members.stream().mapToDouble(NetworkChromosome::getFitness).sum();
        if (totalFitness <= 0) {
            //System.err.println("  -> WARNING: Total fitness is non-positive (" + totalFitness + ") in selectParent(). Selecting randomly.");
            return members.get(new Random().nextInt(members.size()));
        }
    
        double r = new Random().nextDouble() * totalFitness;
        double cumulativeFitness = 0;
    
        for (NetworkChromosome member : members) {
            cumulativeFitness += member.getFitness();
            if (cumulativeFitness >= r) {
                //System.out.println("  -> Selected Parent with Fitness: " + member.getFitness());
                return member;
            }
        }
    
        //System.err.println("  -> ERROR: selectParent() failed to select a valid parent. Selecting randomly.");
        return members.get(new Random().nextInt(members.size()));
    }
    

    public void clear() {
        members.clear();
    }

    public NetworkChromosome getRandomMember() {
        return members.get(new Random().nextInt(members.size()));
    }

    public List<NetworkChromosome> getMembers() {
        return members;
    }

    private double calculateCompatibilityDistance(NetworkChromosome a, NetworkChromosome b) {
        int excess = 0, disjoint = 0;
        double weightDifference = 0;

        for (int i = 0; i < a.getConnections().size(); i++) {
            if (i >= b.getConnections().size()) {
                excess++;
            } else if (a.getConnections().get(i).getInnovationNumber() != b.getConnections().get(i).getInnovationNumber()) {
                disjoint++;
            } else {
                weightDifference += Math.abs(a.getConnections().get(i).getWeight() - b.getConnections().get(i).getWeight());
            }
        }

        return (NEAT.EXCESS_COEFFICIENT * excess + NEAT.DISJOINT_COEFFICIENT * disjoint) / Math.max(a.getConnections().size(), b.getConnections().size())
            + NEAT.WEIGHT_COEFFICIENT * (weightDifference / a.getConnections().size());
    }
}
