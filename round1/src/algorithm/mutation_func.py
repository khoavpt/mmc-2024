import numpy as np

import src.utils.utils as utils

def generate_matrix_with_constraint(dim1, dim2, dim3, num_per_slice_dim3, non_zero_indexes):
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
            p = np.random.randint(0, len(non_zero_indexes))
            i, j = non_zero_indexes[p]
            while (i, j) in assigned_positions:
                i, j = np.random.randint(dim1), np.random.randint(dim2)
            matrix[i, j, k] = 1
            assigned_positions.add((i, j))
    return matrix


def generate_feasible_individual(dataset):
    doctor1_data, doctor2_data, nurse_data = dataset.get_data()
    doc1_non_zero_indexes = np.transpose(np.nonzero(doctor1_data))
    doc2_non_zero_indexes = np.transpose(np.nonzero(doctor2_data))
    nurse_non_zero_indexes = np.transpose(np.nonzero(nurse_data))

    A = generate_matrix_with_constraint(dataset.num_doctor1, dataset.num_faculties, dataset.num_days, dataset.num_doctor1_per_day, doc1_non_zero_indexes)
    B = generate_matrix_with_constraint(dataset.num_doctor2, dataset.num_faculties, dataset.num_days, dataset.num_doctor2_per_day, doc2_non_zero_indexes)
    C = generate_matrix_with_constraint(dataset.num_nurse, dataset.num_faculties, dataset.num_days, dataset.num_nurse_per_day, nurse_non_zero_indexes)
    return utils.matrixes_to_solution(A, B, C, dataset.num_days) # 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days

def generate_feasible_population(dataset, population_size):
    return [generate_feasible_individual(dataset) for _ in range(population_size)]

class CustomMutationFunc:
    def __init__(self, dataset):
        self.dataset = dataset
        self.doctor1_data, self.doctor2_data, self.nurse_data = dataset.get_data()
        
    def custom_mutation_func(self, offspring, ga_instance):
        """
        Args:
            offspring: 2D ndarray of shape (num_offspring, num_genes)
        Returns:
            1D ndarray of shape (num_genes,)
        """
        random_chromosome_idx = np.random.randint(offspring.shape[0])
        offspring[random_chromosome_idx, :] = generate_feasible_individual(dataset=self.dataset)
        return offspring