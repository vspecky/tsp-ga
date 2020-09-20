import sys
from os import path
from random import random, shuffle
from math import floor

CONFIG_INFO = \
"""
There should be a config.tsp file in the directory of the file for configuring the algorithm

Example Config:
    POP_SIZE = 100
    REPR_MODE = cross
    MUT_RATE = 0.05
    GENERATIONS = 100
    VERBOSE = true

The config.tsp file should contain all of the following five keys :-

- POP_SIZE: The size of the population

- REPR_MODE: The mode of reproduction. Can have the following two values:
    1. 'cross' : Reproduction by crossover between two parent Genomes. A randomly chosen
                 consecutive subset of the fitter parent's genes is moved to the child and
                 the rest of the genes are filled in from the less fit parent

    2. 'mut'   : Reproduction by mutation only. A single Genome is copied and the copied
                 child is mutated by swapping the positions of any two randomly chosen genes.

- MUT_RATE: The rate of mutation. Only applicable to Reproduction by Crossover. This is the
                 probability of a child getting mutated and should be between 0 and 1 (inclusive)

- GENERATIONS: The total number of generations for which to run the algorithm.

- VERBOSE: Whether to display verbose output of each generation.

"""

CITIES_INFO = \
"""
There should be a cities.tsp file in the directory of the file containing the cities table.

Example Table:
    A, 0, 1
    B, 0, 2
    C, 0, 3
    D, 0, 4

The format "NAME, X, Y" should be followed and one line should contain one city."
"""

def parse_cities(path):
    with open(path, 'r') as f:
        file_str = f.read().strip()

    rows = file_str.split('\n')

    if len(rows) < 3:
        raise Exception("There should be at least 3 cities")

    cities = {}

    for row in rows:
        city_info = row.split(',')
        if len(city_info) != 3:
            raise Exception(f"Error parsing city info @ '{row}'")

        city_name = city_info[0].strip()
        city_x = float(city_info[1].strip())
        city_y = float(city_info[2].strip())

        cities[city_name] = { 'x': city_x, 'y': city_y }

    if len(cities) < 3:
        raise Exception("There should be at least 3 UNIQUELY NAMED cities")

    return cities

class Progenation:
    Mut = 1
    Crossover = 2

class Gene(object):
    def __init__(self, city_name, city_x, city_y):
        self.name = city_name
        self.x = city_x
        self.y = city_y

    def __add__(self, other):
        if not isinstance(other, Gene):
            raise Exception(f"Can't add a Gene to {type(other)}")

        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __str__(self):
        return f"City({self.name}, ({self.x}, {self.y}))"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Gene):
            return False

        return self.name == other.name

class Genome(object):
    def __init__(self, genes):
        self.genes = genes
        self.calculate_fitness()

    def calculate_fitness(self):
        self.fitness = 0

        for i in range(0, len(self.genes) - 1):
            self.fitness += (self.genes[i] + self.genes[i + 1])

        self.fitness += (self.genes[0] + self.genes[len(self.genes) - 1])

    def mutate(self):
        size = len(self.genes) - 1
        pos1 = floor(random() * size) + 1
        pos2 = floor(random() * size) + 1

        temp = self.genes[pos1]
        self.genes[pos1] = self.genes[pos2]
        self.genes[pos2] = temp

        self.calculate_fitness()

    @staticmethod
    def mutate_child(parent):
        child_genes = parent.genes[:]
        size = len(child_genes) - 1
        pos1 = floor(random() * size) + 1
        pos2 = floor(random() * size) + 1

        temp = child_genes[pos1]
        child_genes[pos1] = child_genes[pos2]
        child_genes[pos2] = temp

    @staticmethod
    def crossover(parent1, parent2):
        male, female = (parent1, parent2) if parent1.fitness > parent2.fitness else (parent2, parent1)

        size = len(male.genes)

        child_genes = [None for _ in range(size)]

        start = floor(random() * size)
        end = floor(random() * size)

        start, end = (start, end) if end > start else (start, end)

        for i in range(start, end + 1):
            child_genes[i] = male.genes[i]

        f_ptr = 0

        for i in range(size):
            if end >= i >= start:
                continue

            gene = None
            for j in range(f_ptr, size):
                f_ptr += 1
                if female.genes[j] not in child_genes:
                    gene = female.genes[j]
                    break

            child_genes[i] = gene

        return Genome(child_genes)

    def __str__(self):
        return f"Genome {{\n    fitness: {self.fitness},\n    path: {' -> '.join([str(g) for g in self.genes])}\n}}"


class TSP_GA(object):
    def __init__(self, file_path, progenation_mode, pop_size, mut_rate=0, verbose=False):
        self.cities = parse_cities(file_path)
        self.progenation_mode = progenation_mode
        self.pop_size = pop_size
        self.verbose = verbose
        self.population = []
        self.mut_rate = mut_rate
        self.genes = [Gene(city, self.cities[city]['x'], self.cities[city]['y']) for city in self.cities.keys()]
        self.reproduction_method = self.reproduce_crossover if progenation_mode == Progenation.Crossover else self.reproduce_mutation
        self.total_fitness = 0
        self.best_genome = None
        self.best_fitness = float('inf')
        self.generations = 0
        self.init_population()

    def init_population(self):
        gene_pool = self.genes[1:]
        origin = self.genes[0]

        for _ in range(self.pop_size):
            shuffle(gene_pool)
            self.population.append(Genome([origin, *gene_pool]))

    def progenate(self):
        self.total_fitness = 0
        for genome in self.population:
            self.total_fitness += genome.fitness

        champ = self.population[0]

        progeny = [self.reproduction_method() for _ in range(self.pop_size - 1)]

        if champ.fitness < self.best_fitness:
            self.best_genome = champ
            self.best_fitness = champ.fitness

        progeny.append(champ)
        self.population = progeny 

        self.generations += 1

        if self.verbose:
            print("=============================================================")
            print(f"Generation: {self.generations}")
            print(f"Current Shortest Path: {' -> '.join([g.name for g in self.best_genome.genes]) + ' -> ' + self.best_genome.genes[0].name}")
            print(f"Shortest Length      : {self.best_fitness}")
            print(f"Average Length       : {self.total_fitness / self.pop_size}")
            print("=============================================================")

    def select_parent(self):
        threshold = random() * self.total_fitness

        rsum = self.total_fitness

        parent = None
        for genome in self.population:
            rsum -= genome.fitness

            if rsum < threshold:
                parent = genome
                break

        if parent == None:
            parent = self.population[0]

        return parent

    def reproduce_crossover(self):
        parent1 = self.select_parent()
        parent2 = self.select_parent()

        child = Genome.crossover(parent1, parent2)

        if random() < self.mut_rate:
            child.mutate()

        return child

    def reproduce_mutation(self):
        parent = self.select_parent()
        return Genome.mutate_child(parent)

def parse_config(path):
    with open(path, 'r') as f:
        file_str = f.read().strip()

    rows = file_str.split('\n')

    if len(rows) != 5:
        raise Exception(CONFIG_INFO)

    config = {}
    
    for row in rows:
        vals = row.split('=')
        key = vals[0].strip()
        value = vals[1].strip()

        if key == "POP_SIZE":
            config[key] = int(value)

        elif key == "REPR_MODE":
            if value in ["cross", "mut"]:
                config[key] = Progenation.Crossover if value == "cross" else Progenation.Mut
            else:
                exc_str = f"Unknown reproduction method '{value}'"
                raise Exception(exc_str)

        elif key == "MUT_RATE":
            try:
                mut_rate = float(value)
            except:
                exc_str = f"Mutation Rate in config.tsp must be a float in the range [0, 1]. Got {value}"
                raise Exception(exc_str)

            if not 1 >= mut_rate >= 0:
                exc_str = f"Mutation Rate in config.tsp must be a float in the range [0, 1]. Got {mut_rate}"
                raise Exception(exc_str)

            config[key] = mut_rate

        elif key == "GENERATIONS":
            exc_str = f"Generations should be a positive integer. Got {value}"
            try:
                gens = int(value)
            except:
                raise Exception(exc_str)

            if gens <= 0:
                raise Exception(exc_str)

            config[key] = gens

        elif key == "VERBOSE":
            if value in ["true", "false"]:
                config[key] = value == "true"
            else:
                exc_str = f"Vebose should be 'true' or 'false'. Got {value}"
                raise Exception(exc_str)

        else:
            exc_str = f"Invalid key found in config.tsp: '{key}'"
            raise Exception(exc_str)

    
    return config


    
def main():
    dir_path = path.dirname(__file__) 
    config = parse_config(path.join(dir_path, "config.tsp"))
    cities_path = path.join(dir_path, "cities.tsp")

    if len(sys.argv) == 2:
        if sys.argv[1] == "help":
            print(CONFIG_INFO)
            print(CITIES_INFO)
            return
        else:
            raise Exception(f"Invalid argument '{sys.argv[1]}'")

    pop = TSP_GA(cities_path, config["REPR_MODE"], config["POP_SIZE"], config["MUT_RATE"], config["VERBOSE"])

    for _ in range(config["GENERATIONS"]):
        pop.progenate() 

    print(f"\nBest Path: {' -> '.join([g.name for g in pop.best_genome.genes]) + ' -> ' + pop.best_genome.genes[0].name}")
    print(f"Distance: {pop.best_fitness}")

if __name__ == "__main__":
    main()
