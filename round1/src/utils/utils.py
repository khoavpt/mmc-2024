import numpy as np

def solution_to_matrixes(solution, num_doctor1, num_doctor2, num_nurse, num_faculties, num_days):
    """
    Args:
        solution: 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days
    Returns:
        A: 3D ndarray of shape (num_doctor1, num_faculties, num_days)
        B: 3D ndarray of shape (num_doctor2, num_faculties, num_days)
        C: 3D ndarray of shape (num_nurse, num_faculties, num_days)
    """
    # A = solution[:self.num_doctor1 * self.num_faculties * self.num_days].reshape(self.num_doctor1, self.num_faculties, self.num_days)
    # B = solution[self.num_doctor1 * self.num_faculties * self.num_days:(self.num_doctor1 + self.num_doctor2) * self.num_faculties * self.num_days].reshape(self.num_doctor2, self.num_faculties, self.num_days)
    # C = solution[(self.num_doctor1 + self.num_doctor2) * self.num_faculties * self.num_days:].reshape(self.num_nurse, self.num_faculties, self.num_days)

    A = solution[:num_doctor1 * num_faculties * num_days].reshape(num_doctor1, num_faculties, num_days)
    B = solution[num_doctor1 * num_faculties * num_days:(num_doctor1 + num_doctor2) * num_faculties * num_days].reshape(num_doctor2, num_faculties, num_days)
    C = solution[(num_doctor1 + num_doctor2) * num_faculties * num_days:].reshape(num_nurse, num_faculties, num_days)

    return A, B, C

def matrixes_to_solution(A, B, C):
    """
    Args:
        A: 3D ndarray of shape (num_doctor1, num_faculties, num_days)
        B: 3D ndarray of shape (num_doctor2, num_faculties, num_days)
        C: 3D ndarray of shape (num_nurse, num_faculties, num_days)
    Returns:
        solution: 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days
    """
    return np.concatenate([A.flatten(), B.flatten(), C.flatten()])
