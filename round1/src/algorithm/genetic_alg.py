import numpy as np
import rootutils
import pygad

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import src.data.dataset as dataset
import src.utils.utils as utils

def generate_matrix_with_constraint(dim1, dim2, dim3, num_per_slice_dim3):
    """
    Args:
        dim1: int
        dim2: int
        dim3: int
        num_per_slice_dim3: int
    Returns:
        3D ndarray of shape (dim1, dim2, dim3)
    """
    matrix = np.zeros((dim1, dim2, dim3))
    for k in range(dim3):
        assigned_positions = set()
        for i in range(num_per_slice_dim3):
            i, j = np.random.randint(dim1), np.random.randint(dim2)
            while (i, j) in assigned_positions:
                i, j = np.random.randint(dim1), np.random.randint(dim2)
            matrix[i, j, k] = 1
            assigned_positions.add((i, j))
    return matrix


def generate_feasible_individual(num_faculties, num_days, num_doctor1, num_doctor2, num_nurse, num_doctor1_per_day, num_doctor2_per_day, num_nurse_per_day):
    A = generate_matrix_with_constraint(num_doctor1, num_faculties, num_days, num_doctor1_per_day)
    B = generate_matrix_with_constraint(num_doctor2, num_faculties, num_days, num_doctor2_per_day)
    C = generate_matrix_with_constraint(num_nurse, num_faculties, num_days, num_nurse_per_day)

    return utils.matrixes_to_solution(A, B, C) # 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days

def generate_feasible_population(population_size, num_faculties, num_days, num_doctor1, num_doctor2, num_nurse, num_doctor1_per_day, num_doctor2_per_day, num_nurse_per_day):
    return [generate_feasible_individual(num_faculties, num_days, num_doctor1, num_doctor2, num_nurse, num_doctor1_per_day, num_doctor2_per_day, num_nurse_per_day) for _ in range(population_size)]

class GeneticAlgorithm:
    def __init__(self, num_genes, fitness_func, num_generations=100, num_parents_mating=10, sol_per_pop=20, parent_selection_type="sss", crossover_type="single_point", mutation_type="random", mutation_percent_genes=10):
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.parent_selection_type = parent_selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mutation_percent_genes = mutation_percent_genes
        self.fitness_func = fitness_func

    def setup(self, initial_population):
        self.ga_instance = pygad.GA(num_generations=self.num_generations,
                                    num_parents_mating=self.num_parents_mating,
                                    fitness_func=self.fitness_func,
                                    num_genes=self.num_genes,
                                    parent_selection_type=self.parent_selection_type,
                                    crossover_type=self.crossover_type,
                                    mutation_type=self.mutation_type,
                                    mutation_percent_genes=self.mutation_percent_genes,
                                    sol_per_pop=self.sol_per_pop,
                                    initial_population=initial_population,
                                    gene_space=[0, 1])

    def run(self):
        print("Start running...")
        self.ga_instance.run()
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print("Run finished!")
        print(f"Fitness value of the best solution: {solution_fitness}")
        return solution, solution_fitness, solution_idx

    def plot_results(self):
        self.ga_instance.plot_fitness()