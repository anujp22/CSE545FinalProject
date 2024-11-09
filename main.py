input_file = "mixed_data.txt"

class BinPackingProblem:
    def __init__(self, capacity, item_weights):
        self.capacity = capacity
        self.item_weights = item_weights

def read_items_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()  
    bin_capacity = int(lines[0].strip().split(":")[1].strip())
    item_weights = [int(line.strip()) for line in lines[1:]]
    return bin_capacity, item_weights

capacity, weights = read_items_file(input_file)

