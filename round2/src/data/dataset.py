import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

MEAL_NUTRITION = {
    'PB': torch.tensor([430, 20, 51, 12, 12, 18]),
    'CF': torch.tensor([280, 2.5, 54, 33.2, 3.3, 11]),
    'Ba': torch.tensor([370, 18, 48, 19, 6, 9]),
}

# Scale the nutrition values to be between 0 and 10
nutrition_matrix = torch.stack(list(MEAL_NUTRITION.values()))

# Normalize column-wise by dividing by the maximum value in each column
max_vals = nutrition_matrix.max(dim=0).values
normalized_matrix = nutrition_matrix / max_vals

# Update MEAL_NUTRITION with normalized values
for i, key in enumerate(MEAL_NUTRITION):
    MEAL_NUTRITION[key] = normalized_matrix[i] * 100

class GlucoseRecordings(Dataset):
    def __init__(self, gv_recordings_path):
        # Load glucose recordings data
        gmr_df = pd.read_csv(gv_recordings_path, sep='\t')

        # Convert 'time' column to datetime
        gmr_df['time'] = pd.to_datetime(gmr_df['time'])

        # Convert 'GlucoseValue' column to numeric, coerce errors to NaN
        gmr_df['GlucoseValue'] = pd.to_numeric(gmr_df['GlucoseValue'], errors='coerce')
        gmr_df = gmr_df.dropna(subset=['GlucoseValue'])
        gmr_df['GlucoseValue'] = gmr_df['GlucoseValue'].astype(float)
        self.gmr_df = gmr_df

        unique_pairs = gmr_df[['userID', 'Meal']].drop_duplicates()
        self.unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))

    def __len__(self):
        return len(self.unique_pairs_list)
    
    def __getitem__(self, idx):
        userID, meal = self.unique_pairs_list[idx]
        user_meal_df = self.gmr_df[(self.gmr_df['userID'] == userID) & (self.gmr_df['Meal'] == meal)]

        t = torch.arange(len(user_meal_df), dtype=torch.float32)
        gv = torch.tensor(user_meal_df['GlucoseValue'].values, dtype=torch.float32)
        meal_nutritions = MEAL_NUTRITION[meal[:2]]

        return t, gv, meal_nutritions

    def get_data_loader(self):
        return DataLoader(self, batch_size=1, shuffle=False)