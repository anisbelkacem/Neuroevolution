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
        neat = new NEAT(10, 10, random); // Initializing with a small population for testing
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
        Agent bestAgent = neat.solve(xorEnv);
        
        assertNotNull(bestAgent, "Best agent should not be null");
        assertTrue(xorEnv.solved(bestAgent), "NEAT should solve the XOR problem");
    }

    @Test
    void testSolveCartPole() {
        Environment cartPoleEnv = getEnvironment(Tasks.CARTPOLE);
        Agent bestAgent = neat.solve(cartPoleEnv);
        
        assertNotNull(bestAgent, "Best agent should not be null");
        assertTrue(bestAgent.getFitness() > 100, "Agent should achieve reasonable fitness in CartPole");
    }

    @Test
    void testMutationAndCrossover() {
        Environment cartPoleEnv = getEnvironment(Tasks.CARTPOLE);
        neat.solve(cartPoleEnv);
        
        assertTrue(neat.getGeneration() > 0, "Mutation and crossover should enable evolutionary progress");
    }
}
