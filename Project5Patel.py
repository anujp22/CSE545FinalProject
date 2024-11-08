import math
import time
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# This function is the implementation of distance formula
def distanceCalculation(x1, y1, x2, y2):
    result = math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
    return result


# This function will read the file and do necessary operations to store the bin capacity and the item sizes in array.
def readingFile(filePath):
    with open(filePath, "r") as file:
        lines = file.readlines()

        # The first line contains the bin capacity
        bin_capacity = int(lines[0].strip())

        # The rest of the lines contain item sizes
        item_sizes = [int(line.strip()) for line in lines[1:] if line.strip()]

    return bin_capacity, item_sizes


# Plot of the improvement of total distance
def plotImprovementCurve(allCurves, title="Improvement Curve", labels=None):
    plt.figure(figsize=(10, 6))

    for i, curve in enumerate(allCurves):
        label = labels[i] if labels else f"Run {i + 1}"
        plt.plot(
            range(1, len(curve) + 1),
            curve,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=5,
            label=label,
        )

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Number of Bins Used")
    plt.grid(True)

    if labels:
        plt.legend()

    plt.show()
    

# 1. Initializing the population
def createIntiliationPopulation(coordinates, sizePopulation):
    population = []
    for _ in range(sizePopulation):
        route = coordinates[1:]
        random.shuffle(route)
        route = [coordinates[0]] + route + [coordinates[0]]
        population.append(route)

    return population


# 2. Fitness Function
def fitness(route):
    totalDistance = 0
    for i in range(len(route) - 1):
        city1 = route[i]
        city2 = route[i + 1]
        totalDistance += distanceCalculation(city1[1], city1[2], city2[1], city2[2])
    return totalDistance


# 3. Selection
def selection(population, fitnesses, tournamentSize):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournamentSize)
        selected.append(min(tournament, key=lambda x: x[1])[0])
    return selected


# 4. Crossover
# 4.1 Ordered Crossover
def orderedCrossover(parent1, parent2):
    start, end = sorted(random.sample(range(1, len(parent1) - 1), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]

    ptr = 0
    for city in parent2:
        if city not in child:
            while ptr < len(child) and child[ptr] is not None:
                ptr += 1
            if ptr < len(child):
                child[ptr] = city

    for i in range(len(child)):
        if child[i] is None:
            for city in parent2:
                if city not in child:
                    child[i] = city
                    break

    child[-1] = child[0]

    return child


# 4.2 Cycle Crossover
def cycleCrossover(parent1, parent2):
    child = [None] * len(parent1)
    visited = [False] * len(parent1)

    index = 0
    while None in child:

        if not visited[index]:
            #
            while not visited[index]:
                visited[index] = True
                child[index] = parent1[index]
                index = parent2.index(parent1[index])
        index += 1

    for i in range(len(child)):
        if child[i] is None:
            child[i] = parent2[i]

    child[-1] = child[0]

    return child


# 5. Heuristic-Based Mutation: Reducing edge crossings
def mutate(route, mutationRate):
    for i in range(1, len(route) - 2):
        if random.random() < mutationRate:

            j = random.randint(i + 1, len(route) - 2)

            segment = route[i : j + 1]

            originalDistance = fitness(route)
            reversedSegment = list(reversed(segment))

            newRoute = route[:i] + reversedSegment + route[j + 1 :]
            newDistance = fitness(newRoute)

            if newDistance < originalDistance:
                route[:] = newRoute

    if random.random() < mutationRate:
        for k in range(len(route) - 3):
            if distanceCalculation(
                route[k][1], route[k][2], route[k + 1][1], route[k + 1][2]
            ) + distanceCalculation(
                route[k + 2][1], route[k + 2][2], route[k + 3][1], route[k + 3][2]
            ) > distanceCalculation(
                route[k][1], route[k][2], route[k + 2][1], route[k + 2][2]
            ) + distanceCalculation(
                route[k + 1][1], route[k + 1][2], route[k + 3][1], route[k + 3][2]
            ):
                # Swap cities to reduce distance
                route[k + 1], route[k + 2] = route[k + 2], route[k + 1]


# 6. Main Evolution with Early Stopping
def geneticAlgorithm(
    coordinates,
    populationSize,
    generation,
    mutationRate,
    tournamentSize,
    crossoverMethod,
    patience=10,
):
    population = createIntiliationPopulation(coordinates, populationSize)

    # Track the best fitness at each generation
    bestFitnessOverTime = []
    best_fitness_current_gen = float(
        "inf"
    )  # Track best fitness of the current generation
    no_improvement_count = 0  # Count generations without improvement

    for gen in range(generation):
        fitnesses = [fitness(route) for route in population]

        # Track the best fitness in this generation
        best_fitness_current_gen = min(fitnesses)
        bestFitnessOverTime.append(best_fitness_current_gen)

        # Check for improvement
        if gen > 0 and best_fitness_current_gen >= bestFitnessOverTime[-2]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0  # Reset if we see an improvement

        if no_improvement_count >= patience:
            print(
                f"Early stopping at generation {gen}. No improvement for {patience} generations."
            )
            break

        selectedPopulation = selection(population, fitnesses, tournamentSize)

        nextGeneration = []
        for i in range(0, len(selectedPopulation), 2):
            parent1 = selectedPopulation[i]
            parent2 = selectedPopulation[i + 1]

            if crossoverMethod == "ordered":
                child1 = orderedCrossover(parent1, parent2)
                child2 = orderedCrossover(parent2, parent1)
            elif crossoverMethod == "cx":
                child1 = cycleCrossover(parent1, parent2)
                child2 = cycleCrossover(parent2, parent1)

            mutate(child1, mutationRate)
            mutate(child2, mutationRate)

            nextGeneration.append(child1)
            nextGeneration.append(child2)

        population = nextGeneration

    bestRoute = min(population, key=lambda route: fitness(route))

    # Return the best route and the improvement curve
    return bestRoute, bestFitnessOverTime


def runExperiments(coordinates, populationSize, generations, tournamentSize):
    crossoverMethods = ["ordered", "cx"]
    mutationRates = [0.01, 0.05]

    bestOverallDistance = float("inf")
    bestMutationRate = None
    bestCrossoverMethod = None
    bestRoute = None
    bestFitnessOverTime = []  # Initialize the best fitness over time list

    results = {}

    allImprovementCurves = []
    curveLabels = []

    for crossover in crossoverMethods:
        for mutationRate in mutationRates:
            key = (
                f"{crossover.capitalize()} Crossover with Mutation Rate {mutationRate}"
            )
            results[key] = []

            # Run the genetic algorithm multiple times to gather results
            for run in range(3):
                print(f"Running {key} - Run {run + 1}...")
                currentBestRoute, improvementCurve = geneticAlgorithm(
                    coordinates,
                    populationSize,
                    generations,
                    mutationRate,
                    tournamentSize,
                    crossover,
                )
                bestDistance = fitness(currentBestRoute)
                results[key].append(bestDistance)  # Store each run's best distance

                if bestDistance < bestOverallDistance:
                    bestOverallDistance = bestDistance
                    bestMutationRate = mutationRate
                    bestCrossoverMethod = crossover
                    bestRoute = currentBestRoute
                    bestFitnessOverTime = (
                        improvementCurve  # Update the best improvement curve
                    )

            allImprovementCurves.append(improvementCurve)
            curveLabels.append(key)

    print(f"Best Overall Distance: {bestOverallDistance}")
    print(f"Best Mutation Rate: {bestMutationRate}")
    print(f"Best Crossover Method: {bestCrossoverMethod}")

    # Return results, best distance, mutation rate, crossover method, route, and the best improvement curve
    return (
        results,
        bestOverallDistance,
        bestMutationRate,
        bestCrossoverMethod,
        bestRoute,
        bestFitnessOverTime,
    )


def analyzeResults(results):
    for key, distances in results.items():
        if len(distances) > 0:  # Check if there are distances
            min_dist = min(distances)
            max_dist = max(distances)
            avg_dist = sum(distances) / len(distances)
            stddev_dist = (
                sum((x - avg_dist) ** 2 for x in distances) / len(distances)
            ) ** 0.5

            print(f"Results for {key}:")
            print(f"Min Distance: {min_dist}")
            print(f"Max Distance: {max_dist}")
            print(f"Average Distance: {avg_dist}")
            print(f"Standard Deviation: {stddev_dist}")
            print("-" * 50)
        else:
            print(f"No distances recorded for {key}.")


# GUI visualization
def animate_route(coordinates, best_route, title="Best Route"):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_title(title, fontsize=16)
    # Extracting x and y coordinates
    x_coords = [city[1] for city in coordinates]
    y_coords = [city[2] for city in coordinates]

    # Preparing the route: Return to the starting city (City 1)
    route_x = [coordinates[int(city[0]) - 1][1] for city in best_route] + [
        coordinates[int(best_route[0][0]) - 1][1]
    ]
    route_y = [coordinates[int(city[0]) - 1][2] for city in best_route] + [
        coordinates[int(best_route[0][0]) - 1][2]
    ]

    # Add padding to the plot to avoid clustering
    x_padding = (max(x_coords) - min(x_coords)) * 0.1
    y_padding = (max(y_coords) - min(y_coords)) * 0.1

    # Set limits for the plot with added padding
    ax.set_xlim(min(x_coords) - x_padding, max(x_coords) + x_padding)
    ax.set_ylim(min(y_coords) - y_padding, max(y_coords) + y_padding)

    # Plot the cities and label them with their numbers
    for i, city in enumerate(coordinates):
        ax.plot(city[1], city[2], "bo")
        ax.text(
            city[1], city[2], f"{int(city[0])}", fontsize=10, ha="right", va="bottom"
        )

    # Highlight the starting/ending city (City 1)
    start_x, start_y = (
        coordinates[int(best_route[0][0]) - 1][1],
        coordinates[int(best_route[0][0]) - 1][2],
    )
    ax.plot(start_x, start_y, "ro", markersize=12)

    # Initialize an empty plot for the route
    (line,) = ax.plot([], [], "bo-", lw=2)

    # Initialize function for animation
    def init():
        line.set_data([], [])
        return (line,)

    # Animation function to update the route
    def update(i):
        line.set_data(route_x[:i], route_y[:i])
        return (line,)

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(route_x), init_func=init, interval=200, blit=True
    )

    plt.show()


def wisdomOfCrowds(population, fitnesses, num_experts=0.4):

    num_experts = max(1, int(len(population) * num_experts))

    expert_indices = np.argsort(fitnesses)[:num_experts]
    experts = [population[i] for i in expert_indices]

    combined_route = experts[0]

    visited = set(combined_route)
    for expert in experts[1:]:
        for city in expert:
            if city not in visited:
                combined_route.append(city)
                visited.add(city)

    combined_route.append(combined_route[0])
    return combined_route


# Modified Genetic Algorithm with Wisdom of Crowds
def hybridGeneticAlgorithm(
    coordinates, populationSize, generation, mutationRate, tournamentSize, patience=10
):
    population = createIntiliationPopulation(coordinates, populationSize)

    bestFitnessOverTime = []
    best_fitness_current_gen = float("inf")
    no_improvement_count = 0

    for gen in range(generation):
        fitnesses = [fitness(route) for route in population]
        best_fitness_current_gen = min(fitnesses)
        bestFitnessOverTime.append(best_fitness_current_gen)

        if gen > 0 and best_fitness_current_gen >= bestFitnessOverTime[-2]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        if no_improvement_count >= patience:
            print(
                f"Early stopping at generation {gen}. No improvement for {patience} generations."
            )
            break

        selectedPopulation = selection(population, fitnesses, tournamentSize)

    nextGeneration = []
    for i in range(0, len(selectedPopulation) - 1, 2):
        parent1 = selectedPopulation[i]
        parent2 = selectedPopulation[i + 1]

        if random.random() < 0.5:
            child1 = orderedCrossover(parent1, parent2)
            child2 = orderedCrossover(parent2, parent1)
        else:
            child1 = cycleCrossover(parent1, parent2)
            child2 = cycleCrossover(parent2, parent1)

        mutate(child1, mutationRate)
        mutate(child2, mutationRate)

        nextGeneration.append(child1)
        nextGeneration.append(child2)

    if len(selectedPopulation) % 2 == 1:
        nextGeneration.append(selectedPopulation[-1])

        combined_route = wisdomOfCrowds(population, fitnesses)
        nextGeneration.append(combined_route)

        population = nextGeneration

    bestRoute = min(population, key=lambda route: fitness(route))
    return bestRoute, bestFitnessOverTime


def main():
    startTime = time.time()
    filePath = "Random97.tsp"
    coordinates = readingFile(filePath)

    configs = [(150, 500, 2), (230, 725, 4), (300, 1000, 5), (200, 300, 3)]

    best_overall_distance_ga = float("inf")
    best_route_ga = None
    best_improvement_curve_ga = None

    best_overall_distance_hybrid = float("inf")
    best_route_hybrid = None
    best_improvement_curve_hybrid = None

    for population_size, generations, tournament_size in configs:
        print(
            f"Running GA with Population Size: {population_size}, Generations: {generations}, Tournament Size: {tournament_size}"
        )
        start_ga = time.time()
        (
            results,
            best_distance,
            mutation_rate,
            crossover_method,
            current_best_route,
            improvement_curve,
        ) = runExperiments(coordinates, population_size, generations, tournament_size)
        run_time_ga = time.time() - start_ga

        if best_distance < best_overall_distance_ga:
            best_overall_distance_ga = best_distance
            best_route_ga = current_best_route
            best_improvement_curve_ga = improvement_curve

    for population_size, generations, tournament_size in configs:
        print(
            f"Running Hybrid GA with Population Size: {population_size}, Generations: {generations}, Tournament Size: {tournament_size}"
        )
        start_hybrid = time.time()
        current_best_route, improvement_curve = hybridGeneticAlgorithm(
            coordinates, population_size, generations, 0.01, tournament_size
        )
        run_time_hybrid = time.time() - start_hybrid
        best_distance = fitness(current_best_route)

        if best_distance < best_overall_distance_hybrid:
            best_overall_distance_hybrid = best_distance
            best_route_hybrid = current_best_route
            best_improvement_curve_hybrid = improvement_curve

    animate_route(coordinates, best_route_ga, title="Best Route of GA")
    animate_route(coordinates, best_route_hybrid, title="Best Route of Hybrid GA")

    plotImprovementCurve([best_improvement_curve_ga], title="GA Improvement Curve")
    plotImprovementCurve(
        [best_improvement_curve_hybrid], title="Hybrid GA Improvement Curve"
    )

    print("=" * 60)
    print("Final Results Summary")
    print("=" * 60)

    print("Genetic Algorithm (GA):")
    print(f"  Best Total Distance: {best_overall_distance_ga:.4f}")
    print(f"  Runtime: {run_time_ga:.4f} seconds")
    print(f"  Best Path: {' -> '.join([str(int(city[0])) for city in best_route_ga])}")
    print("-" * 60)

    print("Hybrid Genetic Algorithm (Hybrid GA):")
    print(f"  Best Total Distance: {best_overall_distance_hybrid:.4f}")
    print(f"  Runtime: {run_time_hybrid:.4f} seconds")
    print(
        f"  Best Path: {' -> '.join([str(int(city[0])) for city in best_route_hybrid])}"
    )

    print("=" * 60)
    print(f"Total Execution Time: {time.time() - startTime:.4f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
