# Neuroevolution: NEAT Implementation

This project is an implementation of the **Neuroevolution of Augmenting Topologies (NEAT)** algorithm, which evolves both the weights and topologies of neural networks using evolutionary strategies instead of traditional gradient descent optimization.

## 📚 Description

NEAT is a powerful genetic algorithm that incrementally evolves neural network structures while maintaining historical context via **innovation numbers**. It uses genetic operations such as **mutation** and **crossover**, along with **speciation**, to evolve neural networks that perform complex tasks like XOR logic and Single-Pole Balancing.

## 🧠 Algorithm Overview

### Gene Encoding

- **Neuron Genes**: Represent input, hidden, output, and bias neurons. Each neuron has properties like type and layer depth.
- **Connection Genes**: Represent directional weighted links between neurons with innovation numbers to track their historical origin.

### Mutation Operators

- **Add Neuron (Structural)**: Inserts a new hidden neuron by splitting a connection.
- **Add Connection (Structural)**: Adds a new forward connection between unconnected neurons.
- **Mutate Weights (Non-Structural)**: Adds Gaussian noise to weights.
- **Toggle Connection (Non-Structural)**: Enables/disables a connection.

### Crossover

- Aligns connection genes by innovation number.
- Inherits matching genes randomly, others from the more fit parent.

### Speciation

- Protects new innovations via compatibility distance `δ`:
- δ = (c1 * E) / N + (c2 * D) / N + c3 * W
  Where:
- `E`: Excess genes
- `D`: Disjoint genes
- `W`: Average weight difference
- `N`: Normalization factor

### Fitness Sharing

Balances species sizes and maintains diversity by dividing individual fitness by species size.

## 🎯 Problem Environments

1. **XOR Function Approximation**
 - Inputs: (0,0), (0,1), (1,0), (1,1)
 - Goal: Evolve a network that models XOR logic.
 - Max fitness: 16

2. **Single-Pole Balancing**
 - Simulates a cart-pole environment.
 - Inputs: Position, Velocity, Angle, Angular Velocity.
 - Output: Force to apply.
 - Fitness: Number of time steps the pole remains balanced.



