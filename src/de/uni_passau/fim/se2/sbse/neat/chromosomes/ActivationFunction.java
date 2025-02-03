package de.uni_passau.fim.se2.sbse.neat.chromosomes;

public enum ActivationFunction {
    NONE {
        @Override
        public double apply(double x) {
            return x; // No activation, just return input as is
        }
    },
    SIGMOID {
        @Override
        public double apply(double x) {
            return 1.0 / (1.0 + Math.exp(-x)); // Standard sigmoid function
        }
    },
    TANH {
        @Override
        public double apply(double x) {
            return Math.tanh(x); // Hyperbolic tangent function
        }
    };

    // Abstract method to be implemented for each activation function type
    public abstract double apply(double x);
}
