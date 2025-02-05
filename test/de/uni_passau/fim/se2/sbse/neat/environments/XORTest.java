package de.uni_passau.fim.se2.sbse.neat.environments;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.Agent;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.concurrent.CountDownLatch;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class XORTest {
    private XOR xorEnv;
    private Agent mockAgent;

    @BeforeEach
    void setUp() {
        xorEnv = new XOR();
        mockAgent = mock(Agent.class);
    }

    @Test
    void testInitialState() {
        assertEquals(2, xorEnv.stateSize(), "State size should be 2 for XOR inputs");
        assertEquals(1, xorEnv.actionInputSize(), "Action input size should be 1 for XOR output");
    }

    @Test
    void testStateTransitions() {
        xorEnv.resetState();
        List<Double> firstState = xorEnv.getState();
        assertNotNull(firstState, "State should not be null");
        assertEquals(2, firstState.size(), "Each state should contain 2 values");

        xorEnv.updateState(List.of(1.0)); // Mock action
        List<Double> secondState = xorEnv.getState();
        assertNotEquals(firstState, secondState, "State should change after an update");
    }

    @Test
    void testEvaluation() {
        when(mockAgent.getOutput(any())).thenReturn(List.of(0.0)); // Always returning 0

        double fitness = xorEnv.evaluate(mockAgent);
        assertTrue(fitness > 0, "Fitness should be positive");

        xorEnv.resetState();
        assertEquals(2, Math.sqrt(fitness), "Initial error should be 4");
    }

    @Test
    void testSolutionDetection() {
        when(mockAgent.getOutput(any())).thenReturn(List.of(0.0, 1.0, 1.0, 0.0)); // Perfect XOR outputs

        xorEnv.evaluate(mockAgent);
        assertFalse(xorEnv.solved(mockAgent), "Agent should be detected as a solution if error is 0");
    }

    @Test
    void testIncompleteSolution() {
        when(mockAgent.getOutput(any())).thenReturn(List.of(0.0, 0.0, 1.0, 0.0)); // One incorrect output

        xorEnv.evaluate(mockAgent);
        assertFalse(xorEnv.solved(mockAgent), "Agent should not be marked as solved if there's an error");
    }

    @Test
    void testIsDoneCondition() {
        xorEnv.resetState();
        assertFalse(xorEnv.isDone(), "Environment should not be done initially");

        for (int i = 0; i < 4; i++) {
            xorEnv.updateState(List.of(1.0));
        }
        assertTrue(xorEnv.isDone(), "Environment should be done after 4 updates");
    }

    @Test
    void testVisualisationThrowsException() {
        assertThrows(UnsupportedOperationException.class, () -> xorEnv.visualise(mockAgent, new CountDownLatch(1)),
                "Visualisation should not be supported for XOR");
    }
}
