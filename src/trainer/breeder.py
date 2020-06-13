from random import sample, random

import numpy as np

from trainer.trainer_config import *


class Breeder:
    NUMBER_NEED_TO_KILL_OFF = int(KILLOFF_RATE * POPULATION)

    def __init__(self, genes):
        self.initialize(genes)
        self.cross_over()
        self.mutate()

    def initialize(self, genes):
        self.genes = genes
        self.genes = list(reversed(sorted(self.genes)))
        self.new_kernels = [_.kernel for _ in self.genes[:POPULATION - self.NUMBER_NEED_TO_KILL_OFF]]

    def cross_over(self):
        for _ in range(self.NUMBER_NEED_TO_KILL_OFF):
            selected_genes = list(sample(self.genes, int(SELECTION_RATE * POPULATION)))
            sorted(selected_genes)
            if selected_genes[0].fitness == 0 and selected_genes[1].fitness == 0:
                self.new_kernels.append(selected_genes[0].kernel)
            else:
                self.new_kernels.append(
                    (selected_genes[0].kernel * selected_genes[0].fitness +
                     selected_genes[1].kernel * selected_genes[1].fitness)
                    / (selected_genes[0].fitness + selected_genes[1].fitness)
                )

    def mutate(self):
        for idx, kernel in enumerate(self.new_kernels):
            if random() > MUTATTION_RATE:
                continue
            self.new_kernels[idx] = kernel + np.random.rand(5, 5) * MUTATION_AMOUNT
