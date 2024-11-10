import random


def generate_data(file_path, bin_capacity=150, num_items=200):
    with open(file_path, "w") as file:
        # Write bin capacity as the first line
        file.write(f"Bin Capacity: {bin_capacity}\n")

        # Generate and write random item weights between 40 and 60
        for _ in range(num_items):
            item_weight = random.randint(20,80)
            file.write(f"{item_weight}\n")


# Generate a file with 200-400 items
generate_data("large2data.txt", bin_capacity=100, num_items=300)
