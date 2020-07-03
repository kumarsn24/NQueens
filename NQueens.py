import random
from typing import List, Any, Union
import sys
import numpy as np
from numpy.random import choice
import json
import pandas as pd


####################################
# File name: NQueens.py            #
# Author: Kumar SN(550484)         #
# Submission: 02-Jul-2020          #
# Version: Python 3.8              #
####################################


class NQueens:

    #
    #    NQueens is a parent class comprising methods to generate population from heredity elements,
    #    Perform variation in population using crossover and mutation parameters and
    #    selects appropriate matched results from total population which ranks higher fitness score.
    #

    def __init__(self, target, total_population, cross_over, mutation_rate):
        # print("Constructor Calling of NQueens")
        self.setup(target, total_population, cross_over, mutation_rate)

    # Function to initialize all parameters required for this genetic algorithm.
    # Function Setup is used to get the target string, total population, cross over and mutation rate parameters used.

    def setup(self, target, total_population, cross_over, mutation_rate):
        print("Initializing Parameters :: Target = %s,%d" % (target, len(target)))
        self.target = target
        self.total_population = total_population
        self.cross_over = cross_over
        self.mutation_rate = mutation_rate
        self.secure_random = random.SystemRandom()
        self.pos_list = [x for x in range(0, 9)]

    # Function view element is used to display individual element under each data object.
    # For instance a[0]=[1,2,3,4]. Each element in data object will be printed using this method.

    def view_element(self, data):
        data = ''.join([elem for elem in str(data)])
        return data

    # Function get Fitness Score is used to calculate fitness score for each combination of population data set
    # This function compares each population element against target state and computes the score
    # Fitness Score : if there is an exact match between population element and target along with their position.

    def get_fitness_score(self, data):
        data = [elem for elem in data]
        fitnessScore1 = 0
        for inloop in range(len(self.target)):
            if int(data[inloop]) == int(self.target[inloop]):
                fitnessScore1 = fitnessScore1 + 1
        return fitnessScore1

    # Function print solution is used to display output position in a matrix format
    # This function compares each population element against target state and updates board position to 1

    def print_solution(self):
        counter = len(self.target)
        i = 1
        columns = 0
        board = [[0] * counter for _ in range(counter)]  # NxN matrix with all elements 0
        for rows in range(len(self.target)):
                try:
                    board[int(self.target[rows:columns+1]) - 1][columns] = 1
                    columns = columns + 1
                except Exception as exc:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

        for values in board:
            print(values)

    # Step 1: Selection - Generate Population Method
    # Create an empty mating pool (an empty Array)
    # For every member of the population, evaluate its fitness based on some criteria / function,
    #      and add it to the population data in a manner consistent with its fitness, i.e. the more fit it
    #      is the more times it appears in the population data, in order to be more likely picked for reproduction.

    populationData = []

    def generate_population(self, target, total_population):
        fitnessData = []
        for outloop in range(total_population):
            randomData = []
            fitnessScore = 0
            for inloop in range(len(target)):
                selectedData = self.secure_random.choice(self.pos_list)
                if selectedData == int(target[inloop]):
                    fitnessScore = fitnessScore + 1
                randomData.append(selectedData)
            self.populationData.append(randomData)
            fitnessData.append(fitnessScore)

        probabilityDist: List[float] = []
        for outloop in range(total_population):
            probabilityDist.append(fitnessData[outloop] / len(target))
        probDataFrame = pd.DataFrame(
            {'Population_Array': self.populationData, 'FitnessScore': fitnessData, 'Probability': probabilityDist})
        probDataFrame = probDataFrame.sort_values(['Probability'], ascending=False)
        probDataFrame = probDataFrame.reset_index(drop=True)
        return probDataFrame

    # Step 2: Perform variation in population by executing the following steps:
    #      1. Pick two "parent" objects from the mating pool.
    #      2. Crossover -- create a "child" object by mating these two parents.
    #      3. Mutation -- mutate the child's data based on a given probability.
    #      4. Add the child object to the new population.
    #      Replace the old population with the new population

    def perform_variation(self, prob_data_frame, generation_count, cross_over, mutation_rate):
        crossOverPoint = int(cross_over * len(self.target))
        print("CrossOver Point : %d" % crossOverPoint, "MutationRate : %f" % mutation_rate)

        for loop in range(generation_count):
            draw = []
            draw.append(prob_data_frame[0:1]["Population_Array"].values[0])
            draw.append(prob_data_frame[1:2]["Population_Array"].values[0])

            if self.get_fitness_score(draw[0]) == len(self.target) | self.get_fitness_score(draw[1]) == len(
                    self.target):
                print("Final Match :", self.view_element(draw[0]), ' ', self.view_element(draw[1]))
                self.print_solution()
                break
            child1 = draw[0][0:crossOverPoint] + draw[1][crossOverPoint:]  # Swap position of elements on cross over
            child2 = draw[1][0:crossOverPoint] + draw[0][crossOverPoint:]

            child1[round(mutation_rate * random.randint(0, len(self.target) - 1))] = self.secure_random.choice(
                self.pos_list)
            child2[round(mutation_rate * random.randint(0, len(self.target) - 1))] = self.secure_random.choice(
                self.pos_list)  # Mutate the population to produce variation.
            self.populationData.append(child1)
            self.populationData.append(child2)
            probabilityDist = []
            fitnessData = []
            total_pops = len(self.populationData)
            for outloop in range(total_pops):
                fitnessScore = self.get_fitness_score(self.populationData[outloop])
                fitnessData.append(fitnessScore)
            for outloop in range(total_pops):
                probabilityDist.append(fitnessData[outloop] / sum(fitnessData))

            prob_data_frame = pd.DataFrame({'Population_Array': self.populationData, 'FitnessScore': fitnessData,
                                            'Probability_Distribution': probabilityDist})
            prob_data_frame = prob_data_frame.sort_values(['Probability_Distribution'], ascending=False)
            prob_data_frame = prob_data_frame.reset_index(drop=True)
            print('Generation ', loop, ' ', ' Average Fitness Score ', prob_data_frame["FitnessScore"].mean(), ' ',
                  ''.join(elem for elem in str(child1)), ':= ', self.get_fitness_score(child1),
                  ''.join(elem for elem in str(child2)),
                  self.get_fitness_score(child2))


mutation_rate = 0.85  # Mutation Rate to reproduce variation.
total_population = 300  # Total Population
cross_over = 0.5  # Cross over percentage
# target = "51863724"  # Target result
generation_count = 1000  # Generation samples


def main():
    if len(sys.argv) < 2:
        print("Input Arguments String/Variable Missing")
        sys.exit()

    # json_string = sys.argv[1]

    try:
        # data = json.loads(str(json_string))
        # NQueens.target = data['qconfig']

        # print ("JSON String:%s" %json_string)

        NQueens.target = sys.argv[1].strip('\"')

        objNQueens = NQueens(NQueens.target, total_population, cross_over, mutation_rate)
        df = objNQueens.generate_population(NQueens.target, total_population)
        objNQueens.perform_variation(df, generation_count, cross_over, mutation_rate)

    except json.JSONDecodeError:
        print('Invalid JSON String format')


if __name__ == "__main__":
    main()
