from random import randint, choices, randrange, random
from collections import namedtuple
import time
import matplotlib.pyplot as plt
import numpy as np

Box = namedtuple('Box', ['name', 'value'])

"""
The closest the distance a box has from a box that we want to be filled, the higher is its value.
distance: 0 => value: 5
distance: 1 => value: 0
distance: 2 => value: 0
distance: 3 => value: 0
distance: 4 => value: 0
distance: 5 => value: 0

In order to calculate the distance between the boxes we use the Manhattan formula.
"""


boxes_K = [
    Box('1', 5),  Box('2', 0),  Box('3', 0),  Box('4', 0),  Box('5', 0),  Box('6', 0),  Box('7', 5),
    Box('8', 5),  Box('9', 0),  Box('10', 0), Box('11', 0), Box('12', 0), Box('13', 5), Box('14', 0),
    Box('15', 5), Box('16', 0), Box('17', 0), Box('18', 0), Box('19', 5), Box('20', 0), Box('21', 0),
    Box('22', 5), Box('23', 0), Box('24', 0), Box('25', 5), Box('26', 0), Box('27', 0), Box('28', 0),
    Box('29', 5), Box('30', 0), Box('31', 5), Box('32', 0), Box('33', 0), Box('34', 0), Box('35', 0),
    Box('36', 5), Box('37', 5), Box('38', 0), Box('39', 0), Box('40', 0), Box('41', 0), Box('42', 0),
    Box('43', 5), Box('44', 0), Box('45', 5), Box('46', 0), Box('47', 0), Box('48', 0), Box('49', 0),
    Box('50', 5), Box('51', 0), Box('52', 0), Box('53', 5), Box('54', 0), Box('55', 0), Box('56', 0),
    Box('57', 5), Box('58', 0), Box('59', 0), Box('60', 0), Box('61', 5), Box('62', 0), Box('63', 0),
    Box('64', 5), Box('65', 0), Box('66', 0), Box('67', 0), Box('68', 0), Box('69', 5), Box('70', 0),
    Box('71', 5), Box('72', 0), Box('73', 0), Box('74', 0), Box('75', 0), Box('76', 0), Box('77', 5)
]

boxes_G = [
    Box('1', 0),  Box('2', 0),  Box('3', 0),  Box('4', 0),  Box('5', 0),  Box('6', 0),  Box('7', 0),
    Box('8', 0),  Box('9', 5),  Box('10', 5), Box('11', 5), Box('12', 5), Box('13', 5), Box('14', 0),
    Box('15', 0), Box('16', 5), Box('17', 0), Box('18', 0), Box('19', 0), Box('20', 0), Box('21', 0),
    Box('22', 0), Box('23', 5), Box('24', 0), Box('25', 0), Box('26', 0), Box('27', 0), Box('28', 0),
    Box('29', 0), Box('30', 5), Box('31', 0), Box('32', 0), Box('33', 0), Box('34', 0), Box('35', 0),
    Box('36', 0), Box('37', 5), Box('38', 0), Box('39', 0), Box('40', 0), Box('41', 0), Box('42', 0),
    Box('43', 0), Box('44', 5), Box('45', 0), Box('46', 0), Box('47', 0), Box('48', 0), Box('49', 0),
    Box('50', 0), Box('51', 5), Box('52', 0), Box('53', 0), Box('54', 0), Box('55', 0), Box('56', 0),
    Box('57', 0), Box('58', 5), Box('59', 0), Box('60', 0), Box('61', 0), Box('62', 0), Box('63', 0),
    Box('64', 0), Box('65', 5), Box('66', 0), Box('67', 0), Box('68', 0), Box('69', 0), Box('70', 0),
    Box('71', 0), Box('72', 5), Box('73', 0), Box('74', 0), Box('75', 0), Box('76', 0), Box('77', 0)
]


def generate_solution():
    """
    Function to generate a possible solution.
    """
    # 1: the box is filled
    # 0: the box is empty
    random_solution = []

    for _ in range(77):
        random_solution.append(randint(0, 1))

    return random_solution


def generate_population(size):  # size represents how many random solutions we want to create
    """
    Function to generate a list of random solutions.

    size: Represents the number of random solutions we are creating
    """
    population = []

    for _ in range(size):
        rnd = generate_solution()
        population.append(rnd)
    
    return population


def fitness(sol, boxes):
    """
    Function to evaluate the solutions.
    """
    filled = 0  # nr of boxes currently filled
    value = 0   

    for i, box in enumerate(boxes):
        if sol[i] == 1:
            filled += 1
            value += box.value

            if filled > 77:
                return 0
    
    return value  # the higher the value, the better the solution is


def select(population, boxes):
    """
    Function to select the solutions to be paired in order to create the next generation.
    """
    possibility = [fitness(sol, boxes) for sol in population]

    # "weights = possibility" means that solutions with a better fitness score have a higher possibility to be selected as parents
    # "k = 2" means that we draw twice from the populations as two parents are needed
    return choices(population = population, weights = possibility, k = 2)


def crossover(sol_a, sol_b):
    """
    Crossover function - defines how two solutions will be paired.
    """
    p = randint(1, 76)  # the length of each solution is 77, so we randomly pick a position between 1 and 76 to cut the solution into two parts
    
    return sol_a[0:p] + sol_b[p:], sol_b[0:p] + sol_a[p:]


def mutation(sol):
    """
    Mutation function - takes as a parameter a possible solution and with a certain probability
    changes 0's to 1's and 1's to 0's in random positions.
    """
    probability = 0.5
    index = randrange(77)
    sol[index] = sol[index] if random() > probability else abs(sol[index] - 1)

    return sol


def run_evolution(boxes, fitness_limit):
    """
    Function that actually runs the evolution, using the predefined methods.
    """
    generation_limit = 10
    population = generate_population(100)

    for _ in range(generation_limit):
        while (fitness(population[0], boxes) != fitness_limit):
            # sorting the current population based to the fitness of each solution so that solutions with better fitness score are first
            population = sorted(
                population,
                key = lambda solution: fitness(solution, boxes),
                reverse = True
                )

            # check if we have reached the fitness_limit, which would mean that the goal is accomplished
            if fitness(population[0], boxes) >= fitness_limit:
                break
    
            # elitism - keep the best two solutions from the previous generation
            next_generation = population[0:2]

            # time to generate all the other new solutions
            for _ in range(int(len(population) / 2) - 1):
                parents = select(population, boxes)
                offspring_a, offspring_b = crossover(parents[0], parents[1])
                offspring_a = mutation(offspring_a)
                offspring_b = mutation(offspring_b)
                next_generation += [offspring_a, offspring_b]
        
            population = next_generation
    
        population = sorted(
            population, 
            key = lambda solution: fitness(solution, boxes),
            reverse = True
        )

    return population


def print_board(sol, boxes):
    """
    Function to print the board with the wanted letter.
    """
    board = np.array(sol)

    for i, box in enumerate(boxes):
        if box.value == 0:
            board[i] = 0

    new_board = board.reshape(11, 7)    # turn the 1d array to 2d
    
    plt.figure(figsize=(4, 4))
    plt.imshow(new_board, cmap='Greys')
    plt.axis(False)
    plt.show()


def choose():
    """
    Function that gives the user the ability to choose the letter he wants to print.
    The user can choose between the letter 'K' and the greek letter 'G'.
    """
    press = input("---CHOOSE THE LETTER YOU WANT TO PRINT. ENTER 'K' OR 'G' IN ORDER TO PRINT THE CORRESPONDING LETTER: ")
    # take only the first letter from the user's input
    press = press[0].lower()

    # check for invalid input
    while press != 'k' and press != 'g':
        press = input("Invalid input! Try again: ")
        press = press[0].lower()
    
    if press == 'k':
        return True
    else: 
        return False


def main():
    """
    Main function.
    """
    choice = choose()
    if choice:
        population = run_evolution(boxes_K, 110)
        print_board(population[0], boxes_K)
    else:
        population = run_evolution(boxes_G, 70)
        print_board(population[0], boxes_G)


if __name__ == "__main__":
    main()
