import csv
import math
import random
import time
import tkinter as tk
from tkinter import scrolledtext


# In bin packing, you're concerned with how to allocate items to bins to minimize the number of bins used or to make the distribution
# as balanced as possible while staying within the bin weight capacity constraints.
# It's a combinatorial optimization problem that has applications in various fields, such as logistics and resource allocation.


# Bin packing class to represent the instance.
class BinPackingProblem:
    def __init__(self, item_weights, truck_weight_capacity):
        self.item_weights = item_weights
        self.truck_weight_capacity = truck_weight_capacity
        # initialze item wieghts and truck weight capacity.

    def __str__(self):
        total_weight = sum(self.item_weights)
        return f"Item Weights in lb: {self.item_weights}\nTruck Weight Capacity: {self.truck_weight_capacity} lb \nTotal Weight of All Items: {total_weight} lb"
        # shows whats the items weights and truck weight capacity are as well as total weight of all weight together.


# individual class to represent a solution
class Individual:
    def __init__(self, item_weights, truck_weight_capacity, mutation_method):
        self.item_weights = item_weights
        self.truck_weight_capacity = truck_weight_capacity
        self.route = []  # Initialize the route (solution)
        self.fitness = 0.0
        self.mutation_method = (
            mutation_method  # Specify the mutation method (swap or scramble)
        )

    def initialize_route(self, num_mutations=4):
        # Initialize the route using a bin packing approach and apply mutations
        total_weight = sum(self.item_weights)
        num_trucks = math.ceil(total_weight / self.truck_weight_capacity)
        bins = [[] for _ in range(num_trucks)]
        random.shuffle(self.item_weights)

        # Randomly shuffle the item weights
        print(f"Total Weight of All Items in lb: {total_weight}")
        for weight in self.item_weights:
            added = False
            for i, bin_items in enumerate(bins):
                if (
                    sum(self.item_weights[item] for item in bin_items) + weight
                    <= self.truck_weight_capacity
                ):
                    bin_items.append(self.item_weights.index(weight))
                    added = True
                    break
            if not added:
                print(
                    f"Truck {self.item_weights.index(weight)} couldn't fit in any bin."
                )

        print("Initial Truck Load:")
        for i, bin_items in enumerate(bins):
            items = [self.item_weights[item] for item in bin_items]
            print(f"Truck {i + 1}: {items} (Total Weight: {sum(items)} Lbs)")

        for _ in range(num_mutations):
            mutation_method = random.choice(["swap", "scramble"])
            print(f"Mutation Method: {mutation_method}")
            self.mutate(mutation_method)

        self.route = bins
        self.calculate_fitness()
        print(f"Fitness: {individual.fitness:.2%}")

        return bins

    def print_route(self):
        print("Initial Route:")
        for i, bin_items in enumerate(self.route):
            items = [self.item_weights[item] for item in bin_items]
            total_weight = sum(items)
            print(f"Truck {i + 1}: {items} (Total Weight: {total_weight}) lb")

    def calculate_fitness(self):
        # Calculate fitness based on how many bins are fully filled
        max_bin_capacity = self.truck_weight_capacity
        total_bins = len(self.route)
        filled_bins = sum(
            1
            for bin_items in self.route
            if sum(self.item_weights[i] for i in bin_items) == max_bin_capacity
        )

        self.fitness = (
            filled_bins / total_bins if total_bins > 0 else 0.0
        )  # Avoid division by zero

    # Mutate the route based on the chosen mutation method
    def mutate(self, mutation_method):
        print(f"Mutation Method: {mutation_method}")

        if mutation_method == "swap":
            self.swap_mutation()
        elif mutation_method == "scramble":
            self.scramble_mutation()

    def swap_mutation(self):
        # Implement swap mutation for Bin Packing Problem
        # Select two random items and swap them between bins
        # It selects two random bins (trucks) within the individual's route.
        # Then, it randomly chooses one item from each of the selected bins.
        # Finally, it swaps these two items between the selected bins.
        if len(self.route) < 2:
            return  # Nothing to swap if there are fewer than 2 bins

        bin1 = random.randint(0, len(self.route) - 1)
        bin2 = bin1
        while bin2 == bin1:
            bin2 = random.randint(0, len(self.route) - 1)

        if self.route[bin1] and self.route[bin2]:
            item1 = random.choice(self.route[bin1])
            item2 = random.choice(self.route[bin2])

            bin1_weight = (
                sum(self.item_weights[i] for i in self.route[bin1])
                - self.item_weights[item1]
                + self.item_weights[item2]
            )
            bin2_weight = (
                sum(self.item_weights[i] for i in self.route[bin2])
                - self.item_weights[item2]
                + self.item_weights[item1]
            )

            if (
                bin1_weight <= self.truck_weight_capacity
                and bin2_weight <= self.truck_weight_capacity
            ):
                self.route[bin1].remove(item1)
                self.route[bin2].remove(item2)
                self.route[bin1].append(item2)
                self.route[bin2].append(item1)
                self.calculate_fitness()

    def scramble_mutation(self):
        # Implement scramble mutation for Bin Packing Problem
        # Select a random bins and scramble the items within it
        if len(self.route) == 0:
            return  # Nothing to scramble if there are no bins

        bin_index = random.randint(0, len(self.route) - 1)
        bin_to_scramble = self.route[bin_index]

        if bin_to_scramble:
            if len(bin_to_scramble) > 1:
                random.shuffle(bin_to_scramble)
                self.route[bin_index] = bin_to_scramble

        self.calculate_fitness()


# Validate if the combined route is a valid solution
def validate_solution(problem_instance, combined_route, truck_capacity):
    all_items = [item for bin_items in combined_route for item in bin_items]

    if set(all_items) != set(problem_instance.item_weights):
        print("Error: Combined route does not contain all original items.")
        return False

    # Sort the bins by their capacity (descending order)
    sorted_bins = sorted(
        enumerate(combined_route),
        key=lambda x: sum(problem_instance.item_weights[item] for item in x[1]),
        reverse=True,
    )

    # Validate the combined route
    total_weight = 0
    for i, (bin_index, bin_items) in enumerate(sorted_bins):
        total_weight += sum(problem_instance.item_weights[item] for item in bin_items)
        if total_weight > truck_capacity:
            return False

    print("Solution is valid.")
    return True


# GUI class to display the results
class BinPackingGUI:
    def __init__(
        self, root, combined_route, individual_stats, statistics, total_execution_time
    ):
        self.root = root
        self.root.title("Bin Packing Solution For Trucks")
        self.combined_route = combined_route
        self.individual_stats = individual_stats
        self.statistics = statistics
        self.total_execution_time = total_execution_time
        # Initialize the GUI with the combined route, individual stats, statistics, and execution time
        self.create_gui()

    def create_gui(self):
        # Display statistics
        individual_stats_label = tk.Label(
            self.root, text="GA Individual Run Statistics on Weight Filling:"
        )
        individual_stats_label.pack()

        individual_stats_text = scrolledtext.ScrolledText(self.root, width=50, height=5)
        individual_stats_text.pack()
        individual_stats_text.insert(tk.END, self.individual_stats)

        # Display detailed combined route
        route_detail_label = tk.Label(
            self.root, text="\nDetailed Combined Truck Weight Routes/Filling:"
        )
        route_detail_label.pack()

        route_detail_text = scrolledtext.ScrolledText(self.root, width=200, height=20)
        route_detail_text.pack()
        route_detail_text.insert(tk.END, self.format_combined_route_detail())

        # Display summary information
        summary_label = tk.Label(
            self.root, text="\nSummary Information for business 100 truck limit:"
        )
        summary_label.pack()

        summary_text = scrolledtext.ScrolledText(self.root, width=200, height=20)
        summary_text.pack()
        summary_text.insert(tk.END, self.format_summary_info())

        # Display remaining weight left over
        remaining_weight_label = tk.Label(
            self.root,
            text=f"\nRemaining Weight Limit and Over Weight Limit {self.calculate_remaining_weight()}",
        )
        remaining_weight_label.pack()

        # Display top combined experts statistics
        stats_label = tk.Label(self.root, text="\nTop Combined Experts Weight Filling:")
        stats_label.pack()

        stats_text = scrolledtext.ScrolledText(self.root, width=50, height=5)
        stats_text.pack()
        stats_text.insert(tk.END, self.statistics)

        # Display total execution time
        execution_time_label = tk.Label(
            self.root,
            text=f"\nTotal Execution Time: {self.total_execution_time:.4f} seconds",
        )
        execution_time_label.pack()

    def format_combined_route_detail(self):
        formatted_route = ""
        for i, bin_items in enumerate(self.combined_route):
            total_weight = sum(bin_items)
            formatted_route += (
                f"Truck {i + 1}: {bin_items} (Total Weight: {total_weight} ton)\n"
            )
        return formatted_route

    def format_summary_info(self):
        max_trucks_to_display = 100
        formatted_summary = "\n"
        for i in range(min(max_trucks_to_display, len(self.combined_route))):
            bin_items = self.combined_route[i]
            total_weight = sum(bin_items)
            formatted_summary += (
                f"Truck {i + 1}: {bin_items} (Total Weight: {total_weight} ton)\n"
            )

        # Calculate and print the remaining weight
        remaining_weight, over_limit_weight = self.calculate_remaining_weight()
        formatted_summary += f"\nRemaining Weight: {remaining_weight} ton\nOver Limit Weight: {over_limit_weight} ton"

        return formatted_summary

    def calculate_remaining_weight(self):
        max_trucks = 100
        max_truck_capacity = 11
        total_assigned_weight = sum(sum(bin_items) for bin_items in self.combined_route)
        remaining_weight = max(
            0, max_truck_capacity * max_trucks - total_assigned_weight
        )
        over_limit_weight = max(
            0, total_assigned_weight - max_truck_capacity * max_trucks
        )
        return remaining_weight, over_limit_weight


# Read input data from a CSV file and create a BinPackingProblem instance
def read_bin_packing_input(file_path):
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            item_weights = [
                int(weight) for weight in row["Item Weight"].strip("[]").split(",")
            ]
            truck_weight_capacity = int(row["Truck Weight Capacity"])
            return BinPackingProblem(item_weights, truck_weight_capacity)


# Specify the path to your input CSV file
input_file = "bin_packing_101.csv"

start_total_time = time.time()

# List of mutation methods
mutation_methods = ["swap", "scramble"]

all_fitness_values = []
num_runs = 100  # Number of generations
total_execution_time = 0
best_individual = None
best_fitness = 0.0
num_loops = 10

# Lists to store the best individuals and their fitness values from each loop
all_best_individuals = []


for loop in range(num_loops):
    print(f"--- Loop {loop + 1} ---")

    # code for one run
    all_fitness_values = []
    best_individual = None
    best_fitness = 0.0

    # Loop through multiple generations
    for run in range(num_runs):
        print(f"--- Truck Run {run + 1} ---")

        # Randomly choose a mutation method
        mutation_method = random.choice(mutation_methods)
        print(f"Mutation Method: {mutation_method}")

        # Read the data and create a BinPackingProblem instance
        problem_instance = read_bin_packing_input(input_file)

        start_time = time.time()
        # Create an Individual instance
        individual = Individual(
            problem_instance.item_weights,
            problem_instance.truck_weight_capacity,
            mutation_method,
        )

        print(problem_instance)

        individual.initialize_route()
        individual.mutate(mutation_method)
        # Initialize the route and apply mutations
        execution_time = time.time() - start_time
        total_execution_time += execution_time

        print("Truck Route after mutation:")
        for i, bin_items in enumerate(individual.route):
            items = [individual.item_weights[item] for item in bin_items]
            total_weight = sum(items)
            print(f"Truck {i + 1}: {items} (Total Weight: {total_weight} lb)")

        # Calculate fitness for the individual
        fitness = sum(
            [
                sum(problem_instance.item_weights[item] for item in bin_items)
                for bin_items in individual.route
            ]
        )
        all_fitness_values.append(
            individual.fitness
        )  # Add this line to append the fitness value to the list
        print(f"Fitness after Mutation: {individual.fitness:.2%}\n")
        print(f"Execution Time: {execution_time:.4f} seconds\n")
        # Update the best individual if the current one has higher fitness
        if individual.fitness > best_fitness:
            best_fitness = individual.fitness
            best_individual = []
            for bin_items in individual.route:
                items = [problem_instance.item_weights[item] for item in bin_items]
                best_individual.append(items)

    print("\nBest Individual:")
    for i, bin_items in enumerate(best_individual):
        total_weight = sum(bin_items)
        print(f"Truck {i + 1}: {bin_items} (Total Weight: {total_weight} lb)")

    print(f"Best Fitness: {best_fitness:.2%}")

    all_best_individuals.append((best_individual, best_fitness))

combined_best_individuals = []
combined_route = []
for i, (individual, fitness) in enumerate(all_best_individuals):
    print(f"\n--- Combined Best Individual {i + 1} ---")
    combined_route.extend(individual)
    for j, bin_items in enumerate(individual):
        total_weight = sum(bin_items)
        print(f"Truck {j + 1}: {bin_items} (Total Weight: {total_weight} lb)")
    print(f"Fitness: {fitness:.2%}")

    combined_best_individuals.append((individual, fitness))

# Calculate the fitness for the combined route
total_fitness_combined = sum(fitness for _, fitness in combined_best_individuals)
fitness_for_combined_route = total_fitness_combined / len(combined_best_individuals)

print("\nCombined Route:")
for i, bin_items in enumerate(combined_route):
    total_weight = sum(bin_items)
    print(f"Truck {i + 1}: {bin_items} (Total Weight: {total_weight} lb)")

print(f"Fitness for Combined Route: {fitness_for_combined_route:.2%}")

average_fitness = sum(all_fitness_values) / num_runs
min_fitness = min(all_fitness_values)
max_fitness = max(all_fitness_values)
std_dev_fitness = math.sqrt(
    sum((x - average_fitness) ** 2 for x in all_fitness_values) / num_runs
)

print("\nStatistics:")
print(f"Average Fitness: {average_fitness:.2%}")
print(f"Min Fitness: {min_fitness:.2%}")
print(f"Max Fitness: {max_fitness:.2%}")
print(f"Standard Deviation of Fitness: {std_dev_fitness:.4f}")

average_fitness = sum(all_fitness_values) / num_runs
min_fitness = min(all_fitness_values)
max_fitness = max(all_fitness_values)
std_dev_fitness = math.sqrt(
    sum((x - average_fitness) ** 2 for x in all_fitness_values) / num_runs
)

average_fitness_combined = (
    sum(fitness for _, fitness in all_best_individuals) / num_loops
)
min_fitness_combined = min(fitness for _, fitness in all_best_individuals)
max_fitness_combined = max(fitness for _, fitness in all_best_individuals)
std_dev_fitness_combined = math.sqrt(
    sum((x - average_fitness_combined) ** 2 for _, x in all_best_individuals)
    / num_loops
)

# Print statistics for the combined best solutions
print("\nStatistics for Combined Best Solutions:")
print(f"Average Fitness: {average_fitness_combined:.2%}")
print(f"Min Fitness: {min_fitness_combined:.2%}")
print(f"Max Fitness: {max_fitness_combined:.2%}")
print(f"Standard Deviation of Fitness: {std_dev_fitness_combined:.4f}")


input_file = "bin_packing_101.csv"
problem_instance = read_bin_packing_input(input_file)


truck_capacity = 11
validate_solution(problem_instance, combined_route, truck_capacity)

print(
    "Original Item Weights:",
    problem_instance.item_weights,
    problem_instance.truck_weight_capacity,
)


max_trucks_to_display = 100

print("\nCombined Route:")
for i in range(min(max_trucks_to_display, len(combined_route))):
    bin_items = combined_route[i]
    total_weight = sum(bin_items)
    print(f"Truck {i + 1}: {bin_items} (Total Weight: {total_weight} lb)")

# Calculate and print the remaining weight
remaining_weight = sum(
    sum(bin_items) for bin_items in combined_route[max_trucks_to_display:]
)
print(f"\nRemaining Weight: {remaining_weight} lb")

total_execution_time = time.time() - start_total_time

# Print total execution time
print(f"\nTotal Execution Time: {total_execution_time:.4f} seconds")

individual_stats = (
    f"Statistics:\n"
    f"Average Fitness: {average_fitness:.2%}\n"
    f"Min Fitness: {min_fitness:.2%}\n"
    f"Max Fitness: {max_fitness:.2%}\n"
    f"Standard Deviation of Fitness: {std_dev_fitness:.4f}"
)


statistics_info = (
    f"Average Fitness: {average_fitness_combined:.2%}\n"
    f"Min Fitness: {min_fitness_combined:.2%}\n"
    f"Max Fitness: {max_fitness_combined:.2%}\n"
    f"Standard Deviation of Fitness: {std_dev_fitness_combined:.4f}"
)


root = tk.Tk()
gui = BinPackingGUI(
    root, combined_route, individual_stats, statistics_info, total_execution_time
)
root.mainloop()
