import numpy as np
import rootutils

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import src.utils.utils as utils

class CustomFitnessFunction:
    def __init__(self, dataset, doc1_max_consecutive_days=2, doc2_max_consecutive_days=2, nurse_max_consecutive_days=2, lambda1=1, lambda2=1, lambda3=1, lambda4=1):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.doc1_max_consecutive_days = doc1_max_consecutive_days
        self.doc2_max_consecutive_days = doc2_max_consecutive_days
        self.nurse_max_consecutive_days = nurse_max_consecutive_days
        self.dataset = dataset
        self.doctor1_data, self.doctor2_data, self.nurse_data = dataset.get_data()

    def soft_constraint_1(self, solution):
        """
        Idea: A day should be covered by all faculties
        => Penalize days that are not covered by all faculties (Add 1 for the number of faculties that are not covered for each day)

        Args:
            solution: 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days
        Returns:
            float
        """
        A, B, C = utils.solution_to_matrixes(solution, self.dataset.num_doctor1, self.dataset.num_doctor2, self.dataset.num_nurse, self.dataset.num_faculties, self.dataset.num_days)
        num_people_per_faculty_per_day = np.sum(np.concatenate([A, B, C], axis=0), axis=0) # shape: (num_faculties, num_days)
        num_faculty_per_day = np.sum(num_people_per_faculty_per_day!=0, axis=0) # shape: (num_days
        num_faculties_not_covered_per_day = self.dataset.num_faculties - num_faculty_per_day # shape: (num_days,)
        violation = np.sum(num_faculties_not_covered_per_day)

        # Min-max normalization for minimization
        min_value = 0  # All faculties are covered everyday
        max_value = (self.dataset.num_faculties - 1) * self.dataset.num_days  # Only one faculty are covered 
        normalized_violation = (violation - min_value) / (max_value - min_value)
        normalized_fitness = 1 - normalized_violation

        return normalized_fitness # float in [0, 1]

    def soft_constraint_2(self, solution):
        """
        Idea: A day should not have too low total years of experience
        => Maximize the total years of experience of people working on all days
        Args:
            solution: 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days
        Returns:
            float
        """
        A, B, C = utils.solution_to_matrixes(solution, self.dataset.num_doctor1, self.dataset.num_doctor2, self.dataset.num_nurse, self.dataset.num_faculties, self.dataset.num_days)
        total_doc1_years = np.sum(np.sum(A, axis=2) * self.doctor1_data) # float
        total_doc2_years = np.sum(np.sum(B, axis=2) * self.doctor2_data)
        total_nurse_years = np.sum(np.sum(C, axis=2) * self.nurse_data)
        fitness = total_doc1_years + total_doc2_years + total_nurse_years # float

        min_value = self.dataset.num_days * (self.dataset.num_doctor1_per_day * np.min(self.doctor1_data) + self.dataset.num_doctor2_per_day * np.min(self.doctor2_data) + self.dataset.num_nurse_per_day * np.min(self.nurse_data))  # Only the least experienced people work
        max_value = self.dataset.num_days * (self.dataset.num_doctor1_per_day * np.max(self.doctor1_data) + self.dataset.num_doctor2_per_day * np.max(self.doctor2_data) + self.dataset.num_nurse_per_day * np.max(self.nurse_data))  # Only the most experienced people work
        normalized_fitness = (fitness - min_value) / (max_value - min_value)

        return normalized_fitness # float in [0, 1]
    
    def soft_constraint_3(self, solution, doc1_max_consecutive_days=2, doc2_max_consecutive_days=2, nurse_max_consecutive_days=2):
        """
        Idea: People should not work on too many consecutive days
        => Penalize if a person works on equal or more than x consecutive days
        Args:
            solution: 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days
            doc1_max_consecutive_days: int
            doc2_max_consecutive_days: int
            nurse_max_consecutive_days: int
        Returns:
            float
        """
        A, B, C = utils.solution_to_matrixes(solution, self.dataset.num_doctor1, self.dataset.num_doctor2, self.dataset.num_nurse, self.dataset.num_faculties, self.dataset.num_days)
        doc1_work_days = np.sum(A, axis=1) # (num_doctor1, num_days)
        doc2_work_days = np.sum(B, axis=1) # (num_doctor2, num_days)
        nurse_work_days = np.sum(C, axis=1) # (num_nurse, num_days)

        violation = (utils.get_number_of_consecutive_sequences_longer_than_k(doc1_work_days, doc1_max_consecutive_days) + utils.get_number_of_consecutive_sequences_longer_than_k(doc2_work_days, doc2_max_consecutive_days) + utils.get_number_of_consecutive_sequences_longer_than_k(nurse_work_days, nurse_max_consecutive_days))
        min_value = 0
        max_value = self.dataset.num_days / 3 * (self.dataset.num_doctor1 + self.dataset.num_doctor2 + self.dataset.num_nurse)
        normalized_violation = (violation - min_value) / (max_value - min_value)
        normalized_fitness = 1 - normalized_violation

        return normalized_fitness # float in [0, 1]

    def soft_constraint_4(self, solution):
        """
        Idea: Doctors and nurses should not work too much or too little
        => Penalize the difference between the maximum and minimum workload of doctors and nurses

        Args:
            solution: 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days
        Returns:
            float
        """
        A, B, C = utils.solution_to_matrixes(solution, self.dataset.num_doctor1, self.dataset.num_doctor2, self.dataset.num_nurse, self.dataset.num_faculties, self.dataset.num_days)
        nurse_workload = np.sum(C, axis=2) # shape: (num_nurse, num_faculties)
        max_nurse_workload_diff = np.max(nurse_workload) - np.min(nurse_workload) # float

        both_doctor_workload = np.sum(np.concatenate([A, B], axis=0), axis=2) # shape: (num_doctor1 + num_doctor2, num_faculties)
        max_doctor_workload_diff = np.max(both_doctor_workload) - np.min(both_doctor_workload) # float
        violation = max_nurse_workload_diff + max_doctor_workload_diff

        min_value = 0  # All have the same workload
        max_value = 2 * self.dataset.num_days # One person works every day
        normalized_violation = (violation - min_value) / (max_value - min_value)
        normalized_fitness = 1 - normalized_violation

        return normalized_fitness # float in [0, 1]

    def violate_hard_constraint(self, solution):
        """
        Idea: Every need to have self.dataset.num_doctor1_per_day doctor1, self.dataset.num_doctor2_per_day doctor2, self.dataset.num_nurse_per_day nurse
        Args:
            solution: 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days
        Returns:
            bool: True if the solution violates the hard constraint, False otherwise
        """
        A, B, C = utils.solution_to_matrixes(solution, self.dataset.num_doctor1, self.dataset.num_doctor2, self.dataset.num_nurse, self.dataset.num_faculties, self.dataset.num_days)
        num_doctor1_per_day = np.sum(A, axis=(0, 1)) # shape: (num_days,)
        num_doctor2_per_day = np.sum(B, axis=(0, 1))
        num_nurse_per_day = np.sum(C, axis=(0, 1))

        return np.any(num_doctor1_per_day != self.dataset.num_doctor1_per_day) or np.any(num_doctor2_per_day != self.dataset.num_doctor2_per_day) or np.any(num_nurse_per_day != self.dataset.num_nurse_per_day)
        
    def custom_fitness_function(self, solution):
        """
        Args:
            solution: 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days
        Returns:
            float
        """
        # return self.lambda1*self.soft_constraint_1(solution) + self.lambda4*self.soft_constraint_4(solution)
        if self.violate_hard_constraint(solution):
            return -np.inf
        return self.lambda1*self.soft_constraint_1(solution) + self.lambda2*self.soft_constraint_2(solution) + self.lambda3*self.soft_constraint_3(solution, self.doc1_max_consecutive_days, self.doc2_max_consecutive_days, self.nurse_max_consecutive_days) +self.lambda4*self.soft_constraint_4(solution)