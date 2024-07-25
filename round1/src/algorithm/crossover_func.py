import numpy as np

class CustomCrossoverFunc:
    def __init__(self, dataset):
        self.dataset = dataset
        self.doctor1_data, self.doctor2_data, self.nurse_data = dataset.get_data()
        
    def custom_crossover_func(self, parents, offspring_size):
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
            
            num_days = self.dataset.num_days
            num_genes_per_day = offspring_size[1] // num_days
            random_split_point = np.random.choice(range(num_days)) * num_genes_per_day

            parent1[random_split_point:] = parent2[random_split_point:]

            offspring.append(parent1)

            idx += 1

        return np.array(offspring)