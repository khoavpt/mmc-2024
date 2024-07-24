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
    return utils.matrixes_to_solution(A, B, C, num_days) # 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days

def generate_feasible_population(population_size, num_faculties, num_days, num_doctor1, num_doctor2, num_nurse, num_doctor1_per_day, num_doctor2_per_day, num_nurse_per_day):
    return [generate_feasible_individual(num_faculties, num_days, num_doctor1, num_doctor2, num_nurse, num_doctor1_per_day, num_doctor2_per_day, num_nurse_per_day) for _ in range(population_size)]


class GeneticAlgorithm:
    def __init__(self, num_genes, fitness_func, dataset, num_generations=100, num_parents_mating=10, sol_per_pop=20, parent_selection_type="sss", mutation_percent_genes=10):
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.parent_selection_type = parent_selection_type
        self.mutation_percent_genes = mutation_percent_genes
        self.fitness_func = fitness_func
        self.dataset = dataset
    @staticmethod
    def custom_crossover_func(parents, offspring_size, ga_instance):
        """
        Args:
            parents: 2D ndarray of shape (num_parents, num_genes)
            offspring_size: tuple of 2 numbers (the offspring size, number of genes).
        """
        offspring = []
        idx = 0
        while len(offspring) != offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            
            num_days = 61
            num_genes_per_day = offspring_size[1] // num_days
            random_split_point = np.random.choice(range(num_days)) * num_genes_per_day

            parent1[random_split_point:] = parent2[random_split_point:]

            offspring.append(parent1)

            idx += 1

        return np.array(offspring)

    @staticmethod
    def custom_mutation_func(offspring, ga_instance):
        """
        Args:
            offspring: 2D ndarray of shape (num_offspring, num_genes)
        Returns:
            1D ndarray of shape (num_genes,)
        """
        random_chromosome_idx = np.random.randint(offspring.shape[0])
        offspring[random_chromosome_idx, :] = generate_feasible_individual(num_faculties=4, num_days=61, num_doctor1=19, num_doctor2=17, num_nurse=44, num_doctor1_per_day=1, num_doctor2_per_day=1, num_nurse_per_day=2)
        return offspring


    def setup(self, initial_population):
        self.ga_instance = pygad.GA(num_generations=self.num_generations,
                                    num_parents_mating=self.num_parents_mating,
                                    fitness_func=self.fitness_func,
                                    num_genes=self.num_genes,
                                    parent_selection_type=self.parent_selection_type,
                                    crossover_type=GeneticAlgorithm.custom_crossover_func,
                                    mutation_type=GeneticAlgorithm.custom_mutation_func,
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