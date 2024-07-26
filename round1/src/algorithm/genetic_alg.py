import numpy as np
import rootutils
import pygad
import matplotlib.pyplot as plt
import seaborn as sns

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

class GeneticAlgorithm:
    def __init__(self, num_genes, fitness_func, crossover_func, mutation_func, dataset, num_generations=100, num_parents_mating=10, sol_per_pop=20, parent_selection_type="sss", mutation_percent_genes=10):
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.parent_selection_type = parent_selection_type
        self.mutation_percent_genes = mutation_percent_genes
        self.fitness_func = fitness_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.dataset = dataset

        self.constraint_evaluations = {'soft_constraint_1': [], 'soft_constraint_2': [], 'soft_constraint_3': [], 'soft_constraint_4': []}
        self.fitness_values = []

    def setup(self, initial_population):
        def fitness_func(ga_instance, solution, solution_idx):
            return self.fitness_func.custom_fitness_function(solution)
        
        def crossover_func(parents, offspring_size, ga_instance):
            return self.crossover_func.custom_crossover_func(parents, offspring_size)

        def mutation_func(offspring, ga_instance):
            return self.mutation_func.custom_mutation_func(offspring, ga_instance)
        
        def callback_function(ga_instance):
            solution, solution_fitness, _ = ga_instance.best_solution()  # Modified to get fitness
            self.fitness_values.append(solution_fitness)  # Store fitness value
            self.constraint_evaluations['soft_constraint_1'].append(self.fitness_func.soft_constraint_1(solution))
            self.constraint_evaluations['soft_constraint_2'].append(self.fitness_func.soft_constraint_2(solution))
            self.constraint_evaluations['soft_constraint_3'].append(self.fitness_func.soft_constraint_3(solution))
            self.constraint_evaluations['soft_constraint_4'].append(self.fitness_func.soft_constraint_4(solution))

        self.ga_instance = pygad.GA(num_generations=self.num_generations,
                                    num_parents_mating=self.num_parents_mating,
                                    fitness_func=fitness_func,
                                    num_genes=self.num_genes,
                                    parent_selection_type=self.parent_selection_type,
                                    crossover_type=crossover_func,
                                    mutation_type=mutation_func,
                                    mutation_percent_genes=self.mutation_percent_genes,
                                    sol_per_pop=self.sol_per_pop,
                                    initial_population=initial_population,
                                    gene_space=[0, 1],
                                    on_generation=callback_function)

    def run(self):
        print("Start running...")
        self.ga_instance.run()
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print("Run finished!")
        print(f"Fitness value of the best solution: {solution_fitness}")
        return solution, solution_fitness, solution_idx

    def plot_total_fitness(self, path=None):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.fitness_values, label='Total Fitness', color='green', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Value')
        plt.title('Total Fitness over Generations')
        plt.legend()
        if path:
            plt.savefig(path)
        plt.show()

    def plot_constraints(self, path=None):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        for key, value in self.constraint_evaluations.items():
            sns.lineplot(data=value, label=key)
        plt.xlabel('Generation')
        plt.ylabel('Value')
        plt.title('Constraints over Generations')
        plt.legend()
        if path:
            plt.savefig(path)
        plt.show()