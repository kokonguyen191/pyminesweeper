class Gene:
    def __init__(self, kernel, fitness=0):
        self.kernel = kernel
        self.fitness = fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __lt__(self, other):
        return self.fitness < other.fitness
