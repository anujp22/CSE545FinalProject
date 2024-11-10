import random
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

# Dataset details
inputFile = "uniformData.txt"


class BinPackingProblem:
    def __init__(self, capacity, itemWeights):
        self.capacity = capacity
        self.itemWeights = itemWeights


def readItemsFile(filePath):
    with open(filePath, "r") as f:
        lines = f.readlines()
    binCapacity = int(lines[0].strip().split(":")[1].strip())
    itemWeights = [int(line.strip()) for line in lines[1:]]
    return binCapacity, itemWeights


binCapacity, items = readItemsFile(inputFile)


# Genetic Algorithm Classes and Functions
class Individual:
    @staticmethod
    def fitness(solution, binCapacity):
        bins = []
        currentBin = []
        currentBinWeight = 0

        for itemSize in solution:
            if currentBinWeight + itemSize <= binCapacity:
                currentBin.append(itemSize)
                currentBinWeight += itemSize
            else:
                bins.append(currentBin)
                currentBin = [itemSize]
                currentBinWeight = itemSize

        if currentBin:
            bins.append(currentBin)  # Add the last bin if it contains any items

        return len(bins), bins


def createInitialPopulation(items, populationSize, binCapacity):
    population = [random.sample(items, len(items)) for _ in range(populationSize)]
    return population


def selection(population, fitnesses, tournamentSize):
    # Ensure tournamentSize is not larger than the population size
    tournamentSize = min(tournamentSize, len(population))
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournamentSize)
        selected.append(min(tournament, key=lambda x: x[1])[0])
    return selected


def crossover(parent1, parent2, binCapacity):
    # Simple one-point crossover
    midpoint = len(parent1) // 2
    child = parent1[:midpoint] + parent2[midpoint:]
    return child


def mutate(solution, binCapacity, mutationRate=0.1):
    # Mutate by swapping two random items
    for _ in range(int(len(solution) * mutationRate)):
        i, j = random.sample(range(len(solution)), 2)
        solution[i], solution[j] = solution[j], solution[i]


def wisdomOfCrowds(population, fitnesses, binCapacity, numExperts=0.4):
    numExperts = max(1, int(len(population) * numExperts))
    expertIndices = np.argsort(fitnesses)[:numExperts]
    experts = [population[i] for i in expertIndices]
    combinedSolution = []
    for expert in experts:
        for item in expert:
            if item not in combinedSolution:
                combinedSolution.append(item)
    return combinedSolution


# Main GA Function with GUI Integration
class BinPackingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bin Packing Problem - GA with Wisdom of Crowds")
        self.geometry("800x600")

        # Frame for results display
        self.resultsFrame = tk.Frame(self)
        self.resultsFrame.pack(pady=10)

        # Run button
        self.runButton = tk.Button(
            self, text="Run Algorithm", command=self.runAlgorithm
        )
        self.runButton.pack(pady=20)

        # Label for best solution display
        self.bestSolutionLabel = tk.Label(
            self.resultsFrame, text="", font=("Arial", 12)
        )
        self.bestSolutionLabel.pack()

    def runAlgorithm(self):
        populationSize = 50
        generations = 100
        mutationRate = 0.05
        tournamentSize = min(3, populationSize)

        bestSolution, improvementCurve = hybridGeneticAlgorithmBPP(
            items,
            binCapacity,
            populationSize,
            generations,
            mutationRate,
            tournamentSize,
        )

        bestBinCount, binsUsed = Individual.fitness(bestSolution, binCapacity)

        # Display best solution in the GUI
        self.bestSolutionLabel.config(
            text=f"Best Solution Uses {bestBinCount} Bins\nBest Solution: {binsUsed}"
        )

        # Plot the improvement curve in the GUI
        self.plotImprovementCurve(improvementCurve)

        # Print the results to the terminal
        self.printResults(bestSolution, improvementCurve)

    def plotImprovementCurve(self, improvementCurve):
        plt.figure(figsize=(8, 6))
        plt.plot(improvementCurve, marker="o", color="b", label="Bins Used")
        plt.title("Improvement Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Number of Bins Used")
        plt.grid(True)
        plt.legend()
        plt.show()

    def printResults(self, bestSolution, improvementCurve):
        # Print the best solution and improvement curve to the terminal
        bestBinCount, binsUsed = Individual.fitness(bestSolution, binCapacity)
        print(f"Best Solution Uses {bestBinCount} Bins")
        print(f"Best Solution (Bins): {binsUsed}")

        print("\nPacking Summary:")
        print(f"The solution uses {bestBinCount} bins to pack the items.")
        print("Bin contents (items in each bin):")
        for i, bin in enumerate(binsUsed, 1):
            print(f"  Bin {i}: {bin}")

        print("\nImprovement Overview:")
        print(
            "The algorithm progressively reduced the number of bins used over the course of the run."
        )
        print("Final packing solution:")
        print(f"  Uses {bestBinCount} bins.")
        print("Packing details:")
        for i, bin in enumerate(binsUsed, 1):
            print(f"  Bin {i}: {bin}")


def hybridGeneticAlgorithmBPP(
    items,
    binCapacity,
    populationSize,
    generations,
    mutationRate,
    tournamentSize,
    patience=10,
):
    population = createInitialPopulation(items, populationSize, binCapacity)
    bestFitnessOverTime = []
    noImprovementCount = 0
    bestSolution = None
    bestFitness = float("inf")

    for gen in range(generations):
        fitnesses = [
            Individual.fitness(solution, binCapacity)[0] for solution in population
        ]
        currentBestFitness = min(fitnesses)
        bestFitnessOverTime.append(currentBestFitness)

        if currentBestFitness < bestFitness:
            bestFitness = currentBestFitness
            bestSolution = population[fitnesses.index(bestFitness)]
            noImprovementCount = 0
        else:
            noImprovementCount += 1

        if noImprovementCount >= patience:
            break

        selectedPopulation = selection(population, fitnesses, tournamentSize)
        nextGeneration = []

        for i in range(0, len(selectedPopulation) - 1, 2):
            parent1, parent2 = selectedPopulation[i], selectedPopulation[i + 1]
            child = crossover(parent1, parent2, binCapacity)
            mutate(child, binCapacity, mutationRate)
            nextGeneration.append(child)

        combinedSolution = wisdomOfCrowds(population, fitnesses, binCapacity)
        nextGeneration.append(combinedSolution)
        population = nextGeneration

    return bestSolution, bestFitnessOverTime


if __name__ == "__main__":
    app = BinPackingGUI()
    app.mainloop()
