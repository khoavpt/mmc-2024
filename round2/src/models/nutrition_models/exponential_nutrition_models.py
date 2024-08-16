import torch
import torch.nn as nn

class ExponentialNutritionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize parameters for each nutrition (calories, fat, carb, sugar, fiber, protein)
        self.t_meal = nn.Parameter(torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=torch.float32))
        
    def forward(self, t, meal_nutritions):
        """
        Args:
            t: (meal_records_length)
            meal_nutritions: (6)
        """
        t = t.unsqueeze(1) # (meal_records_length, 1)
        result = torch.sum(meal_nutritions * torch.exp(-t/self.t_meal), dim=1) # (meal_records_length, 1)
        return result # (meal_records_length, 1)