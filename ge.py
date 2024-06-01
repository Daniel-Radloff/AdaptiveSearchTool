import random as rand
import pandas
import numpy as np
import math
import sys
import timeit
from copy import deepcopy


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    per = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    if count == total:
        sys.stdout.write('[%s] %s%s ...%s\n' % (bar, per, '%', suffix))
    else:
        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, per, '%', suffix))
        sys.stdout.flush()


GENERATION_SIZE = 50
POPULATION_SIZE = 50
TOURNAMENT_SIZE = 4
INDIVIDUAL_SIZE = 30

MAX_NUM_EXPR = 15

CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.5

DATASET_TRAIN = './Data/Train_cleaned.csv'
DATASET_TEST = './Data/Test_cleaned.csv'
DATA_ds = pandas.read_csv(DATASET_TRAIN)
TEST_DATA_ds = pandas.read_csv(DATASET_TEST)
DATA = DATA_ds.to_dict('records')
TEST_DATA = TEST_DATA_ds.to_dict('records')



COLUMNS = DATA_ds.columns
COLUMNS = COLUMNS.drop('attack')
# print(COLUMNS)



grammar = {
    'expr': [('var',), ('expr', 'bi_func', 'var'), ('var', 'bi_func', 'expr'), ('un_func', 'expr')],
    'bi_func': ['+', '-', '*'],
    'un_func': ['sin', 'cos', 'tan', 'sqrt', 'abs', 'log'],
    'var':  [('terminal',), ('const',)],
    'terminal': COLUMNS.values,
    'const': [i/10.0 for i in range(-50, 50)]
}

def divide(x, y):
    if y == 0:
        return 0
    else:
        return x / y

def mod(x, y):
    if y == 0:
        return 0
    else:
        return x % y

def sqrt(x):
    if x < 0:
        return math.sqrt(-x)
    else:
        return math.sqrt(x)

def log(x):
    if x <= 0:
        return math.log(0.0001)
    else:
        return math.log(x)

function_map = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'abs': abs,
    'sqrt': sqrt,
    'log': log,
}

class gp:

    def __init__(self):
        self.population = []
        self.population_fitness = []
        self.generation_best = []
        self.indiv_index = 0
        self.best_fitness = 0
        self.best_indiv = []
        self.train_size = len(DATA)
        self.test_size = len(TEST_DATA)
        self.initialise()
        self.train()
        # self.predict_test(self.best_indiv)

    def initialise(self):
        for _ in range(POPULATION_SIZE):
            self.population.append(self.generate_individual(INDIVIDUAL_SIZE))
    
    def train(self):
        for i in range(GENERATION_SIZE):
            # progress(i, GENERATION_SIZE-1)
            self.evolve()

    def predict_test(self, grammar):
        correct = 0
        TruePositives = 0
        FalsePositives = 0
        FalseNegatives = 0
        # print(grammar)
        compiled_grammar = compile(grammar, '<string>', 'eval')
        # grammar = self.get_indiv_grammer(indiv)
        for row in TEST_DATA:
            outcome = row['attack']


            try:
                value = math.tanh(self.calculate_grammar(compiled_grammar, row))
            except:
                value = 0

            if value > 0 and outcome == 1:
                correct += 1
                TruePositives += 1
            elif value > 0 and outcome == 0:
                FalsePositives += 1
            elif value <= 0 and outcome == 0:
                correct += 1
            elif value <= 0 and outcome == 1:
                FalseNegatives += 1

        accuracy = correct / self.test_size
        recall = TruePositives / (TruePositives + FalseNegatives)
        precision = TruePositives / (TruePositives + FalsePositives)
        f_score = 2 * (precision * recall) / (precision + recall)

        # print(f'Test Data Accuracy: {accuracy}')
        # print(f'Recall: {recall}')
        # print(f'Precision: {precision}')
        # print(f'F-Score: {f_score}')

        return {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F-Score': f_score}

    def get_indiv_fitness(self, grammar):
        correct = 0
        compiled_grammar = compile(grammar, '<string>', 'eval')

        for row in DATA:
            # outcome = getattr(row, 'attack')
            outcome = row['attack']
            # value = self.calculate_grammar(grammar, row)
            # value = 1 / (1 + np.exp(-self.calculate_grammar(grammar, row)))

            # np.tanh(1)
            try:
                value = math.tanh(self.calculate_grammar(compiled_grammar, row))
            except:
                value = 0

            if value > 0 and outcome == 1:
                correct += 1
            elif value <= 0 and outcome == 0:
                correct += 1

        return correct / self.train_size

    def calculate_grammar(self, compiled_grammar, context):

        try:
            result = eval(compiled_grammar, function_map, context)

            if result == None:
                print("here")
                print(compiled_grammar)
            return result
        except ZeroDivisionError:
            print("here")
            return 0



    def get_indiv_grammar(self, indiv, stack = [], expr = MAX_NUM_EXPR):
        if stack == []:
            expr_stack = ['expr']
            self.indiv_index = 0
        elif expr == 0:
            expr_stack = ['var']
        else:
            expr_stack = stack

        rec_stack = []
        output = []

        while expr_stack:
            curr = expr_stack.pop()

            curr = grammar[curr][indiv[self.indiv_index] % len(grammar[curr])]

            self.indiv_index += 1

            if self.indiv_index >= len(indiv):
                self.indiv_index = 0

            if isinstance(curr, tuple):
                if curr == ('un_func', 'expr'):
                    rec_stack.extend(('expr',))
                    un_func = grammar['un_func'][indiv[self.indiv_index] % len(grammar['un_func'])]
                    curr = un_func + '(' + self.get_indiv_grammar(indiv, rec_stack, expr - 1) + ')'
                else:
                    rec_stack.extend(reversed(curr))
                    curr = self.get_indiv_grammar(indiv, rec_stack, expr - 1)
            
            output.append(str(curr))

        if len(output) > 1:
            return '(' + ''.join(output) + ')'
        else:
            return output[0]

    def generate_individual(self, length):
        array = [rand.randint(0, 100) for _ in range(length)]
        while array[0] % len(grammar['expr']) == 0:
            array[0] = rand.randint(0, 100)

        return array

    def calcualte_population_fittness(self):
        max_fitness = 0
        max_indiv = []
        self.population_fitness = []
        for indiv in self.population:
            grammer = self.get_indiv_grammar(indiv)
            fitness = self.get_indiv_fitness(grammer)
            self.population_fitness.append(fitness)
            if fitness > max_fitness:
                max_fitness = fitness
                max_indiv = indiv
        curr_grammar = self.get_indiv_grammar(max_indiv)

        if max_fitness > self.best_fitness:
            # print(max_fitness)
            self.best_fitness = max_fitness
            self.best_indiv = curr_grammar
        # print(self.population_fitness)
        self.generation_best.append(max_indiv)
        return (max_fitness, curr_grammar)

    def write_to_file(self):
        with open('Transfer/mendeley.txt', 'w') as f:
            for indiv in self.population:
                array_str = ' '.join(map(str, indiv))
                f.write(f'{array_str} \n')

    def tournament_selection(self):
        best = None
        max_fitness = 0

        for _ in range(TOURNAMENT_SIZE):
            index = rand.randint(0, POPULATION_SIZE - 1)
            fitness = self.population_fitness[index]
            # grammer = self.get_indiv_grammer(self.population[index])
            # fitness = self.get_indiv_fitness(grammer)

            if fitness >= max_fitness:
                max_fitness = fitness
                best = self.population[index]

        return best

    def crossover(self):
        parent1 = deepcopy(self.tournament_selection())
        parent2 = deepcopy(self.tournament_selection())
        crossover_point1 = rand.randint(1, len(parent1))
        crossover_point2 = rand.randint(1, len(parent2))

        child1 = parent1[:crossover_point1] + parent2[crossover_point2:]
        child2 = parent2[:crossover_point2] + parent1[crossover_point1:]

        child1 = self.prune(child1)
        child2 = self.prune(child2)

        return (child1, child2)
    
    def mutate(self):
        parent =  deepcopy(self.tournament_selection())

        for i in range(len(parent)):
            if rand.uniform(0, 1) < MUTATION_RATE:
                parent[i] = rand.randint(0, 100)
                if i == 0:
                    while parent[i] % len(grammar['expr']) == 0:
                        parent[i] = rand.randint(0, 100)

        return parent
    
    def prune(self, indiv):
        array = indiv
        if len(array) > INDIVIDUAL_SIZE:
                array = array[:INDIVIDUAL_SIZE]  # Remove the excess elements from the start

        # print(array) 
        return array


    def evolve(self):

        new_population = []

        # self.calcualte_population_fittness()
        print(self.calcualte_population_fittness())

        for _ in range(math.floor(POPULATION_SIZE * CROSSOVER_RATE)):
            new_nodes = self.crossover()
            new_population.append(new_nodes[0])
            new_population.append(new_nodes[1])

        #Mutation
        for _ in range(POPULATION_SIZE - len(new_population)):
            new_population.append(self.mutate())



        self.population = new_population


def run():
    seeds = [989, 796, 451, 565, 7, 92, 932, 1234, 961, 826]
    # seeds = [989, 796, 451]
    print(seeds)
    data = pandas.DataFrame({'Seed': [], 'Accuracy': [], 'Recall': [], 'Precision': [], 'F-Score': [], 'runtime': []})
    for seed in seeds:
        print(f'Seed: {seed}')
        rand.seed(seed)


        start = timeit.default_timer()

        Gp = gp()


        # Gp.train()

        stop = timeit.default_timer()

        result = Gp.predict_test(Gp.best_indiv)
        # result = Gp.predict()

        result['Seed'] = seed

        result['runtime'] = stop - start

        print(result)

        data = pandas.concat([data, pandas.DataFrame(result, index=[0])], ignore_index=True)
    
    print(data)
    # data.to_csv('mendeley_results.csv', index=False)
    print(f'Average Accuracy: {data["Accuracy"].mean()}, Max Accuracy: {data["Accuracy"].max()}')
    print(f'Average Recall: {data["Recall"].mean()}, Max Recall: {data["Recall"].max()}')
    print(f'Average Precision: {data["Precision"].mean()}, Max Precision: {data["Precision"].max()}')
    print(f'Average F-Score: {data["F-Score"].mean()}, Max F-Score: {data["F-Score"].max()}')
    print(f'Average Runtime: {data["runtime"].mean()}, Max Runtime: {data["runtime"].max()}')



if __name__ == "__main__":

    Gp = gp()
    # Gp.write_to_file()
    # run()