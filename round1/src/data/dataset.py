import pandas as pd
import rootutils

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

def read_to_numpy(file_path):
    df = pd.read_csv(file_path, index_col=False).reset_index()
    return df.to_numpy()

class Dataset:
    def __init__(self, doctor1_file_path, doctor2_file_path, nurse_file_path, num_doctor1_per_day=1, num_doctor2_per_day=1, num_nurse_per_day=2, num_faculties=4, num_days=61):
        self.doctor1_data = read_to_numpy(ROOTPATH / doctor1_file_path)
        print(ROOTPATH / doctor1_file_path)
        self.doctor2_data = read_to_numpy(ROOTPATH / doctor2_file_path)
        self.nurse_data = read_to_numpy(ROOTPATH / nurse_file_path)

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