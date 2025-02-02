package de.uni_passau.fim.se2.sbse.neat.algorithms.innovations;

/**
 * Concrete implementation of the Innovation interface.
 */
public class InnovationImpl implements Innovation {

    private final int innovationNumber;

    

    /**
     * Creates a new innovation with a unique innovation number.
     *
     * @param innovationNumber The unique ID assigned to this innovation.
     */
    public InnovationImpl(int innovationNumber) {
        this.innovationNumber = innovationNumber;
    }
    
    public int getInnovationNumber() {
        return innovationNumber;
    }

    @Override
    public int hashCode() {
        return Integer.hashCode(innovationNumber);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        InnovationImpl that = (InnovationImpl) obj;
        return innovationNumber == that.innovationNumber;
    }

    @Override
    public String toString() {
        return "Innovation{" + "innovationNumber=" + innovationNumber + '}';
    }
}
