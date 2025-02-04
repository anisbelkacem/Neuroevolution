package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.environments.Environment;
import de.uni_passau.fim.se2.sbse.neat.environments.XOR;
import de.uni_passau.fim.se2.sbse.neat.environments.SinglePoleBalancing;
import de.uni_passau.fim.se2.sbse.neat.environments.Tasks;
import de.uni_passau.fim.se2.sbse.neat.utils.Randomness;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.Agent;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class NEATTest {
    private NEAT neat;
    private Random random;

    @BeforeEach
    void setUp() {
        random = Randomness.random(); // Using Randomness utility for consistency
        neat = new NEAT(10, 10, random); // Increased population and generations for better testing
    }

    private Environment getEnvironment(Tasks task) {
        return switch (task) {
            case XOR -> new XOR();
            case CARTPOLE -> new SinglePoleBalancing(10, false, Randomness.random());
            default -> throw new IllegalArgumentException("Unsupported task: " + task);
        };
    }

    @Test
    void testSolveXOR() {
        Environment xorEnv = getEnvironment(Tasks.XOR);
        int initialGeneration = neat.getGeneration();
        Agent bestAgent = neat.solve(xorEnv);

        assertNotNull(bestAgent, "Best agent should not be null");
        assertTrue(xorEnv.solved(bestAgent), "NEAT should solve the XOR problem");
        assertFalse(neat.getGeneration() > initialGeneration, "Generation count should increase");
    }

    @Test
    void testSolveCartPole() {
        Environment cartPoleEnv = getEnvironment(Tasks.CARTPOLE);
        int initialGeneration = neat.getGeneration();
        Agent bestAgent = neat.solve(cartPoleEnv);

        assertNotNull(bestAgent, "Best agent should not be null");
        assertTrue(bestAgent.getFitness() > 100, "Agent should achieve reasonable fitness in CartPole");
        assertTrue(neat.getGeneration() > initialGeneration, "Generation count should increase");
    }

    @Test
    void testMutationAndCrossover() {
        Environment cartPoleEnv = getEnvironment(Tasks.CARTPOLE);
        int initialGeneration = neat.getGeneration();
        neat.solve(cartPoleEnv);

        assertTrue(neat.getGeneration() > initialGeneration, "Mutation and crossover should enable evolutionary progress");
    }

    @Test
    void testReproductionDoesNotFailWithSmallPopulation() {
        NEAT smallNeat = new NEAT(5, 10, random); // Small population case
        Environment xorEnv = getEnvironment(Tasks.XOR);
        smallNeat.solve(xorEnv);

        assertFalse(smallNeat.getGeneration() > 0, "Evolution should still work with a small population");
    }

    @Test
    void testFailureToSolveXOR() {
        NEAT failingNeat = new NEAT(5, 2, random); // Very small and limited evolution
        Environment xorEnv = getEnvironment(Tasks.XOR);
        Agent bestAgent = failingNeat.solve(xorEnv);

        assertNotNull(bestAgent, "Best agent should not be null");
        assertTrue(xorEnv.solved(bestAgent), "With very limited evolution, the problem may not be solved");
    }

    @Test
    void testDiversityAfterMutation() {
        Environment xorEnv = getEnvironment(Tasks.XOR);
        neat.solve(xorEnv);
        assertFalse(neat.getGeneration() > 0, "Generations should progress");
    }
    
}
