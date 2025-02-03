package de.uni_passau.fim.se2.sbse.neat.algorithms;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkChromosome;
import java.util.*;

public class Species {
    private final List<NetworkChromosome> members;
    private NetworkChromosome representative;
    private double adjustedFitness;
    private double compatibilityThreshold = 3.0; // Initial speciation threshold

    public Species(NetworkChromosome representative) {
        this.representative = representative;
        this.members = new ArrayList<>();
        this.members.add(representative);
    }

    /**
     * Computes the genetic distance between this species and a given network.
     * The formula: δ = (c1 * D / N) + (c2 * E / N) + (c3 * W̄)
     * Where:
     * - D = Number of disjoint genes
     * - E = Number of excess genes
     * - W̄ = Average weight difference of matching genes
     */
    public double computeDistance(NetworkChromosome other, double c1, double c2, double c3) {
        int matchingGenes = 0;
        int disjointGenes = 0;
        int excessGenes = 0;
        double weightDifferenceSum = 0.0;

        Map<Integer, Double> parentWeights = new HashMap<>();
        for (var connection : representative.getConnections()) {
            parentWeights.put(connection.getInnovationNumber(), connection.getWeight());
        }

        int maxInnovationRep = representative.getConnections().stream()
                .mapToInt(c -> c.getInnovationNumber()).max().orElse(0);
        int maxInnovationOther = other.getConnections().stream()
                .mapToInt(c -> c.getInnovationNumber()).max().orElse(0);
        int maxInnovation = Math.max(maxInnovationRep, maxInnovationOther);

        Map<Integer, Double> otherWeights = new HashMap<>();
        for (var connection : other.getConnections()) {
            otherWeights.put(connection.getInnovationNumber(), connection.getWeight());
        }

        for (var innovation : parentWeights.keySet()) {
            if (otherWeights.containsKey(innovation)) {
                matchingGenes++;
                weightDifferenceSum += Math.abs(parentWeights.get(innovation) - otherWeights.get(innovation));
            } else {
                
                disjointGenes++;
            }
        }

        for (var innovation : otherWeights.keySet()) {
            if (!parentWeights.containsKey(innovation)) {
                disjointGenes++;
            }
        }

        excessGenes = maxInnovation - Collections.max(parentWeights.keySet());

        int N = Math.max(representative.getConnections().size(), other.getConnections().size());
        if (N < 20) N = 1; 

        double avgWeightDiff = (matchingGenes > 0) ? (weightDifferenceSum / matchingGenes) : 0.0;

        return ((c1 * disjointGenes) / N) + ((c2 * excessGenes) / N) + (c3 * avgWeightDiff);
    }

    /**
     * Determines if a given network should belong to this species.
     */
    public boolean belongsToSpecies(NetworkChromosome network) {
        return computeDistance(network, 1.0, 1.0, 0.4) < compatibilityThreshold;
    }

    /**
     * Adds a new network to this species.
     */
    public void addMember(NetworkChromosome network) {
        members.add(network);
    }

    /**
     * Clears members except the representative, preparing for a new generation.
     */
    public void reset() {
        members.clear();
        members.add(representative);
    }

    /**
     * Computes the adjusted fitness of the species.
     */
    public void computeAdjustedFitness() {
        double totalFitness = members.stream().mapToDouble(NetworkChromosome::getFitness).sum();
        adjustedFitness = totalFitness / members.size(); // Fitness sharing
    }

    /**
     * Returns the best performing network in this species.
     */
    public NetworkChromosome getBestNetwork() {
        return members.stream().max(Comparator.comparingDouble(NetworkChromosome::getFitness)).orElse(representative);
    }

    
    public NetworkChromosome reproduce() {
        members.sort(Comparator.comparingDouble(NetworkChromosome::getFitness).reversed());
        return members.get(0); 
    }

    public List<NetworkChromosome> getMembers() {
        return members;
    }

    public NetworkChromosome getRepresentative() {
        return representative;
    }

    public void setRepresentative(NetworkChromosome representative) {
        this.representative = representative;
    }

    public double getAdjustedFitness() {
        return adjustedFitness;
    }
}
