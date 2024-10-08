import numpy as np

def matrixes_to_solution(A, B, C, num_days):
    """
    Args:
        A: 3D ndarray of shape (num_doctor1, num_faculties, num_days)
        B: 3D ndarray of shape (num_doctor2, num_faculties, num_days)
        C: 3D ndarray of shape (num_nurse, num_faculties, num_days)
    Returns:
        solution: 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days
    """
    A = A.reshape(-1, num_days)
    B = B.reshape(-1, num_days)
    C = C.reshape(-1, num_days)
    merged = np.concatenate([A, B, C], axis=0).swapaxes(0, 1)
    return merged.flatten()

def solution_to_matrixes(solution, num_doctor1, num_doctor2, num_nurse, num_faculties, num_days):
    """
    Args:
        solution: 1D ndarray of shape (num_doctor1 + num_doctor2 + num_nurse) * num_faculties * num_days
        num_doctor1: Number of doctors in group 1
        num_doctor2: Number of doctors in group 2
        num_nurse: Number of nurses
        num_faculties: Number of faculties
        num_days: Number of days
    Returns:
        A: 3D ndarray of shape (num_doctor1, num_faculties, num_days)
        B: 3D ndarray of shape (num_doctor2, num_faculties, num_days)
        C: 3D ndarray of shape (num_nurse, num_faculties, num_days)
    """
    solution = solution.reshape(num_days, -1).swapaxes(0, 1)
    endA = num_doctor1 * num_faculties
    endB = endA + num_doctor2 * num_faculties

    A = solution[:endA].reshape(num_doctor1, num_faculties, num_days)
    B = solution[endA:endB].reshape(num_doctor2, num_faculties, num_days)
    C = solution[endB:].reshape(num_nurse, num_faculties, num_days)

    return A, B, C

def get_number_of_consecutive_sequences_longer_than_k(arr, k):
    """
    Calculate the number of consecutive sequences longer than length k in axis 1
    Args:
        arr: binary 2D ndarray
        k: int
    Returns:
        int
    """
    zeros = np.zeros((1, arr.shape[1]))
    arr = np.concatenate([zeros, arr, zeros], axis=0)
    arr = arr.swapaxes(1, 0).flatten()
    diff = np.diff(arr)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    lengths = ends - starts
    return np.sum(lengths > 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

