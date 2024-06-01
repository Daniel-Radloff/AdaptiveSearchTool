from abc import abstractmethod
import sys
sys.path.append('../Library/')
import random
import pandas
import math
from copy import deepcopy
import timeit

import cProfile
import pstats
from io import StringIO

DATASET_TRAIN = 'Data/Train_cleaned.csv'
DATASET_TEST = 'Data/Test_cleaned.csv'
DATA_ds = pandas.read_csv(DATASET_TRAIN)
DATA_ds = DATA_ds.sample(frac=0.1, random_state=1)
TEST_DATA_ds = pandas.read_csv(DATASET_TEST)
DATA = DATA_ds.to_dict('records')
TEST_DATA = TEST_DATA_ds.to_dict('records')


COLUMNS = DATA_ds.columns
COLUMNS = COLUMNS.drop('attack')

POPULATION_SIZE = 50
GENERATIONS = 50
MAX_TREE_DEPTH = 10
TOURNAMENT_SIZE = 3


CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.3
# REPRODUCTION_RATE = 0.1

FUNCTION_SET = COLUMNS.values
TERMINAL_SET = ['1', '0']


class Node:
    def __init__(self,type: str, level: int) -> None:
        self.type = type
        self.level = level
        self.fitness = 0
        self.left = None
        self.right = None

    def print(self, tabs = 0) -> None:
        print('\t' * tabs,  self.type)
        if self.right != None:
            self.right.print(tabs + 1)
        if self.left != None:
            self.left.print(tabs + 1)

    @abstractmethod
    def toString(self, string) -> None:
        pass
        
    

class FunctionNode(Node):

    def __init__(self, type: str, level: int, min: float, max: float) -> None:
        Node.__init__(self, type, level)
        self.min = min
        self.max = max
        self.predictValue = random.uniform(self.min,self.max)

    def predict(self, data) -> None:
        if data[self.type] < self.predictValue:
            return self.left.predict(data)
        else:
            return self.right.predict(data)

    def toString(self) -> None:
        string = self.type + '-' + str(self.level) + '-' + str(self.predictValue) +'('
        string += self.left.toString()
        string += ')('
        string += self.right.toString()
        string += ')'
        return string


class TerminalNode(Node):
    def predict(self, data: tuple) -> None:
        return self.type
    
    def toString(self) -> None:
        return str(self.type) + '-' + str(self.level)

class GP:
    def __init__(self) -> None:
        print("Gp started")
        # random.seed(989)
        self.population = []

        self.columns_min = {}
        self.columns_max = {}
        self.get_max_of_columns(DATA_ds)
        self.get_min_of_columns(DATA_ds)

        self.training_size = len(DATA)
        self.global_best = None
        self.avg_fitness = 0

    def get_min_of_columns(self, data) -> None:
        for column in COLUMNS.values:
            self.columns_min[column] = data[column].min()

    def get_max_of_columns(self, data) -> None:
        for column in COLUMNS.values:
            self.columns_max[column] = data[column].max()

    def grow(self, depth: int, max: int) -> None:
        # Choose terminal but make sure it is not head of table
        if(depth == 0 or random.uniform(0, 1) > 0.3 and depth < max):
            return TerminalNode(random.randint(0,1), depth)

        # type = FUNCTION_SET[random.randint(0, 7)]
        type = FUNCTION_SET[random.randint(0, 10)]
        curr_node = FunctionNode(type, depth, self.columns_min[type], self.columns_max[type])

        curr_node.left = self.grow(depth - 1, max)
        curr_node.right = self.grow(depth - 1, max)

        return curr_node

    def full(self, depth) -> None:
        # Choose terminal but make sure it is not head of table
        if(depth == 0):
            return TerminalNode(random.randint(0,1), depth)
        else:
            type = FUNCTION_SET[random.randint(0, 10)]
            # type =  FUNCTION_SET[random.randint(0, 7)]
            curr_node = FunctionNode(type, depth, self.columns_min[type], self.columns_max[type])

            curr_node.left = self.full(depth - 1)
            curr_node.right = self.full(depth - 1)

            return curr_node
    
    def initialize(self) -> None:

        batch_size = math.floor(POPULATION_SIZE / (MAX_TREE_DEPTH))

        #Each Level
        for level in range(1, MAX_TREE_DEPTH):
            # Half Grow
            for _ in range(math.floor(batch_size / 2)):
                self.population.append(self.grow(level, level))
            #Half Full
            for _ in range(math.ceil(batch_size / 2)):
                self.population.append(self.full(level))

        #Do Max depth
        batch_size = POPULATION_SIZE - (batch_size * (MAX_TREE_DEPTH - 1))
        
        # Half Grow
        for _ in range(math.floor(batch_size / 2)):
            self.population.append(self.grow(MAX_TREE_DEPTH, MAX_TREE_DEPTH))
        #Half Full
        for _ in range(math.ceil(batch_size / 2)):
            self.population.append(self.full(MAX_TREE_DEPTH))


    def train(self, single = False) -> None:

        self.initialize()

        for _ in range(GENERATIONS):
           self.evolve(single)

        # Best population fitness eval
        for node in self.population:
            node.fitness = self.treeFitness(node)


    def predict(self, single = False) -> None:
        # test_data = pandas.read_csv(DATASET_TEST)

        test_size = len(TEST_DATA)

        correct = 0
        TruePositives = 0
        FalsePositives = 0
        FalseNegatives = 0
        node = self.getWinner()


        for row in TEST_DATA:
            if node.predict(row) == row['attack']: 
                correct += 1
                if row['attack'] == 1:
                    TruePositives += 1
            else:
                if row['attack'] == 0:
                    FalseNegatives += 1
                else:
                    FalsePositives += 1

        accuracy = correct / test_size
        recall = TruePositives / (TruePositives + FalseNegatives)
        precision = TruePositives / (TruePositives + FalsePositives)
        f_score = 2 * (precision * recall) / (precision + recall)

        if single:
            print(f'Test Data Accuracy: {accuracy}')
            print(f'Recall: {recall}')
            print(f'Precision: {precision}')
            print(f'F-Score: {f_score}')

        return {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F-Score': f_score}

    def treeFitness(self, node: Node) -> None:
        correct = 0

        for row in DATA:
            if node.predict(row) == row['attack']: 
                correct += 1

        return correct / self.training_size

    def tournamentSelection(self) -> None:

        tournament: list = []
        for i in range(TOURNAMENT_SIZE):
            tournament.append(self.population[random.randint(0, POPULATION_SIZE - 1)])

        max = 0

        for i in range(1, TOURNAMENT_SIZE):
            if tournament[i].fitness > tournament[max].fitness:
                max = i

        return tournament[max]

    def crossover(self) -> None:
        parent1 = deepcopy(self.tournamentSelection())
        parent2 = deepcopy(self.tournamentSelection())

        subtree1 = parent1
        subtree2 = parent2

        subtree1dir = "left"
        subtree2dir = "left"

        curr = parent1

        for i in range(random.randint(0, MAX_TREE_DEPTH)):
            if random.randint(0, 1) == 0:
                if curr.left != None:
                    subtree1dir = "left"
                    subtree1 = curr
                    curr = curr.left
                else:
                    break
            else:
                if curr.right != None:
                    subtree1dir = "right"
                    subtree1 = curr
                    curr = curr.right
                else:
                    break

        curr = parent2

        for i in range(random.randint(0, MAX_TREE_DEPTH)):
            if random.randint(0, 1) == 0:
                if curr.left != None:
                    subtree2dir = "left"
                    subtree2 = curr
                    curr = curr.left
                else:
                    break
            else:
                if curr.right != None:
                    subtree2dir = "right"
                    subtree2 = curr
                    curr = curr.right
                else:
                    break

        if subtree1dir == "left":
            temp = subtree1.left
            if subtree2dir == "left":
                subtree1.left = subtree2.left
                subtree2.left = temp
            else:
                subtree1.left = subtree2.right
                subtree2.right = temp


        else:
            temp = subtree1.right
            if subtree2dir == "left":
                subtree1.right = subtree2.left
                subtree2.left = temp
            else:
                subtree1.right = subtree2.right
                subtree2.rigth = temp

        self.trim_tree(parent1)
        self.trim_tree(parent2)

        return (parent1, parent2)

    def permute(self) -> None:
        parent = deepcopy(self.tournamentSelection())

        subtree = parent
        curr = parent

        for i in range(random.randint(0, MAX_TREE_DEPTH)):
            if random.randint(0, 1) == 0:
                if curr.left != None:
                    subtree = curr
                    curr = curr.left
                else:
                    break
            else:
                if curr.right != None:
                    subtree = curr
                    curr = curr.right
                else:
                    break

        temp = subtree.left
        subtree.left = subtree.right
        subtree.right = temp
        
        return parent 

    def mutate(self) -> None:
        parent = deepcopy(self.tournamentSelection())

        subtree = parent
        subtreedir = "left"

        curr = parent

        for i in range(random.randint(0, MAX_TREE_DEPTH)):
            if random.randint(0, 1) == 0:
                if curr.left != None:
                    subtree = curr
                    subtreedir = "left"
                    curr = curr.left
                else:
                    break
            else:
                if curr.right != None:
                    subtree = curr
                    subtreedir = "right"
                    curr = curr.right
                else:
                    break

        if subtreedir == "left":
            subtree.left = self.grow(random.randint(0, MAX_TREE_DEPTH), MAX_TREE_DEPTH)
        else:
            subtree.right = self.grow(random.randint(0, MAX_TREE_DEPTH), MAX_TREE_DEPTH)

        self.trim_tree(parent)
        
        return parent
    
    def trim_tree(self, node, depth = 0):
        if node.left != None and node.right != None:
            if depth + 1 >= MAX_TREE_DEPTH:
                node.left = TerminalNode(random.randint(0,1), 0)
                node.right = TerminalNode(random.randint(0,1), 0)
            else:
                if node.left != None:
                    self.trim_tree(node.left, depth + 1)
                if node.right != None:
                    self.trim_tree(node.right, depth + 1)

    def evolve(self, single = False) -> None:

        # Get fitness of each tree
        for node in self.population:
            node.fitness = self.treeFitness(node)
        
        if single:
            print("Best:", self.getWinner().fitness)

        new_population: list = []

        # Crossover
        for _ in range(math.floor(POPULATION_SIZE * CROSSOVER_RATE / 2)):
            new_nodes = self.crossover()
            new_population.append(new_nodes[0])
            new_population.append(new_nodes[1])

        #Reproduction
        # for _ in range(math.floor(POPULATION_SIZE * REPRODUCTION_RATE)):
        #     new_population.append(deepcopy(self.tournamentSelection()))

        #Mutation
        for _ in range(POPULATION_SIZE - len(new_population)):
            new_population.append(self.mutate())

        self.population = new_population
            
    def getWinner(self) -> None:

        if self.global_best == None:
            self.global_best = self.population[0]

        max = self.population[0]
        for node in self.population:
            if node.fitness > max.fitness:
                if node.fitness >= self.global_best.fitness:
                    self.global_best = node
                max = node

        self.avg_fitness += max.fitness
        return max

    def sortPopulation(self) -> None:
        self.population.sort(key=lambda x: x.fitness, reverse=True)
    
    def appendToFile(self) -> None:
        print(self.global_best.toString())
        # self.global_best.print()
        self.sortPopulation()
        with open('./Results/mendeley.txt', 'w') as f:
            # f.write(self.global_best.toString())
            # f.write('\n')
            for tree in self.population:
                f.write(tree.toString())
                f.write('\n')



def run():
    seeds = [989, 796, 451, 565, 7, 92, 932, 1234, 961, 826]
    # seeds = [989, 796, 451]
    print(seeds)
    data = pandas.DataFrame({'Seed': [], 'Accuracy': [], 'Recall': [], 'Precision': [], 'F-Score': [], 'runtime': []})
    for seed in seeds:
        print(f'Seed: {seed}')
        random.seed(seed)
        Gp = GP()
        start = timeit.default_timer()

        Gp.train()

        stop = timeit.default_timer()

        result = Gp.predict()

        result['Seed'] = seed

        result['runtime'] = stop - start

        print(result)

        data = pandas.concat([data, pandas.DataFrame(result, index=[0])], ignore_index=True)

    print(data)
    data.to_csv('mendeley_results.csv', index=False)
    print(f'Average Accuracy: {data["Accuracy"].mean()}, Max Accuracy: {data["Accuracy"].max()}')
    print(f'Average Recall: {data["Recall"].mean()}, Max Recall: {data["Recall"].max()}')
    print(f'Average Precision: {data["Precision"].mean()}, Max Precision: {data["Precision"].max()}')
    print(f'Average F-Score: {data["F-Score"].mean()}, Max F-Score: {data["F-Score"].max()}')
    print(f'Average Runtime: {data["runtime"].mean()}, Max Runtime: {data["runtime"].max()}')



if __name__ == "__main__":


    Gp = GP()
    profiler = cProfile.Profile()
    profiler.enable()

    Gp.train(True)

    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)  # Print top 10 results sorted by cumulative time
    print(s.getvalue())
    

    # Gp.predict(True)


