import pandas as pd
import rootutils
import numpy as np

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

def read_to_numpy(file_path):
    """
    Args:
        file_path: Path to the csv file
    Returns:
        year_data: 2D ndarray of shape (num_rows, num_faculties)
                   where year_data[i, j] = years of experience of ith person in jth faculty if exists, 0 otherwise
    """
    data = pd.read_csv(file_path, index_col=False).to_numpy()
    num_faculties = np.max(data[:, 1]) + 1
    year_data = np.zeros((data.shape[0], num_faculties))
    for i in range(data.shape[0]):
        year_data[i, data[i, 1]] = data[i, 0]
    return year_data # 2D ndarray of shape (num_rows, num_faculties)

class Dataset:
    def __init__(self, doctor1_file_path, doctor2_file_path, nurse_file_path, num_doctor1_per_day=1, num_doctor2_per_day=1, num_nurse_per_day=2, num_faculties=4, num_days=61):
        self.doctor1_data = read_to_numpy(ROOTPATH / doctor1_file_path) # 2D ndarray of shape (num_doctor1, num_faculties)
        self.doctor2_data = read_to_numpy(ROOTPATH / doctor2_file_path) # 2D ndarray of shape (num_doctor2, num_faculties)
        self.nurse_data = read_to_numpy(ROOTPATH / nurse_file_path) # 2D ndarray of shape (num_nurse, num_faculties)

        self.num_doctor1 = self.doctor1_data.shape[0] 
        self.num_doctor2 = self.doctor2_data.shape[0]
        self.num_nurse = self.nurse_data.shape[0]

        self.num_doctor1_per_day = num_doctor1_per_day
        self.num_doctor2_per_day = num_doctor2_per_day
        self.num_nurse_per_day = num_nurse_per_day

        self.num_faculties = num_faculties
        self.num_days = num_days
    
    def get_data(self):
        return self.doctor1_data, self.doctor2_data, self.nurse_data