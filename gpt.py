import random
import time
import numpy as np
import matplotlib.pyplot as plt


# Fitness Function: Calculates the number of bins used for a solution
def fitness(solution, bin_capacity):
    bins = []
    for item_size in solution:
        placed = False
        for bin in bins:
            if sum(bin) + item_size <= bin_capacity:
                bin.append(item_size)
                placed = True
                break
        if not placed:
            bins.append([item_size])
    return len(bins)  # The fewer bins used, the better


# Create Initial Population for BPP
def create_initial_population(items, population_size, bin_capacity):
    population = []
    for _ in range(population_size):
        # Shuffle items to create a random assignment
        solution = random.sample(items, len(items))
        population.append(solution)
    return population


# Tournament Selection
def selection(population, fitnesses, tournament_size):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        selected.append(min(tournament, key=lambda x: x[1])[0])
    return selected


# Bin-Packing Crossover
def crossover(parent1, parent2, bin_capacity):
    # Simple crossover by combining halves of each parent
    midpoint = len(parent1) // 2
    child = parent1[:midpoint] + parent2[midpoint:]

    # Ensure bins do not exceed capacity
    child_bins = []
    current_bin = []
    for item in child:
        if sum(current_bin) + item > bin_capacity:
            child_bins.append(current_bin)
            current_bin = [item]
        else:
            current_bin.append(item)
    child_bins.append(current_bin)
    return child


# Mutation: Randomly reassign an item to a different bin
def mutate(solution, bin_capacity, mutation_rate=0.1):
    for _ in range(int(len(solution) * mutation_rate)):
        i, j = random.sample(range(len(solution)), 2)
        solution[i], solution[j] = solution[j], solution[i]


# Wisdom of Crowds Aggregation
def wisdom_of_crowds(population, fitnesses, bin_capacity, num_experts=0.4):
    num_experts = max(1, int(len(population) * num_experts))
    expert_indices = np.argsort(fitnesses)[:num_experts]
    experts = [population[i] for i in expert_indices]

    # Combine the bins of expert solutions
    combined_solution = []
    for expert in experts:
        for item in expert:
            if item not in combined_solution:
                combined_solution.append(item)
    return combined_solution


# Main Genetic Algorithm with Wisdom of Crowds for BPP
def hybrid_genetic_algorithm_bpp(
    items,
    bin_capacity,
    population_size,
    generations,
    mutation_rate,
    tournament_size,
    patience=10,
):
    population = create_initial_population(items, population_size, bin_capacity)
    best_fitness_over_time = []
    no_improvement_count = 0
    best_solution = None
    best_fitness = float("inf")

    for gen in range(generations):
        fitnesses = [fitness(solution, bin_capacity) for solution in population]
        current_best_fitness = min(fitnesses)

        # Track fitness improvement
        best_fitness_over_time.append(current_best_fitness)

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[fitnesses.index(best_fitness)]
            no_improvement_count = 0  # Reset if improved
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(
                f"Early stopping at generation {gen}. No improvement for {patience} generations."
            )
            break

        # Select, crossover, and mutate
        selected_population = selection(population, fitnesses, tournament_size)
        next_generation = []

        for i in range(0, len(selected_population) - 1, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child = crossover(parent1, parent2, bin_capacity)
            mutate(child, bin_capacity, mutation_rate)
            next_generation.append(child)

        # Wisdom of Crowds aggregation
        combined_solution = wisdom_of_crowds(population, fitnesses, bin_capacity)
        next_generation.append(combined_solution)
        population = next_generation

    return best_solution, best_fitness_over_time


# Run the algorithm
def main():
    # Example bin packing items and capacity
    bin_capacity = 10
    items = [2, 5, 4, 7, 1, 3, 8, 5, 6]
    population_size = 50
    generations = 100
    mutation_rate = 0.05
    tournament_size = 3

    best_solution, improvement_curve = hybrid_genetic_algorithm_bpp(
        items,
        bin_capacity,
        population_size,
        generations,
        mutation_rate,
        tournament_size,
    )

    print(f"Best solution found uses {fitness(best_solution, bin_capacity)} bins.")
    print("Solution:", best_solution)

    # Plot the improvement curve
    plt.plot(improvement_curve, marker="o")
    plt.title("Improvement Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Number of Bins Used")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
