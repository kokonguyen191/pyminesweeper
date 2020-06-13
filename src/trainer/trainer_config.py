POPULATION = 1000
NUMBER_OF_GENERATIONS = 100
TURN_NUMBER_LIMIT = -1  # Set to -1 for unlimited
NO_OF_THREADS = 250  # MUST DIVIDE POPULATION!
NEW_TRAINING_SESSION = True
STARTING_GENERATION = 0

# BREEDER SETTINGS
SELECTION_RATE = 0.1  # Two best genes from SELECTION_RATE * population will be chosen to cross over
KILLOFF_RATE = 0.3  # KILLOFF_RATE * POP = no. of genes killed off
MUTATTION_RATE = 0.1  # The probability of a gene getting a mutation
MUTATION_AMOUNT = 2  # The max amount of mutation allowed
