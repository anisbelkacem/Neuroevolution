package de.uni_passau.fim.se2.sbse.neat.algorithms;

import java.util.*;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkChromosome;

class Species {
    private final List<NetworkChromosome> members;
    private final NetworkChromosome representative;

    public Species(NetworkChromosome representative) {
        this.members = new ArrayList<>();
        this.representative = representative;
        members.add(representative);
    }

    public void addMember(NetworkChromosome chromosome) {
        members.add(chromosome);
    }

    public NetworkChromosome getRepresentative() {
        return representative;
    }

    public List<NetworkChromosome> getMembers() {
        return members;
    }
}