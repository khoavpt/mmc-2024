import torch
import torch.nn as nn

class GammaNutritionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize parameters for each nutrition (calories, fat, carb, sugar, fiber, protein)
        self.k = nn.Parameter(torch.tensor([8.0, 8.0, 8.0, 8.0, 8.0, 8.0], dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32))
    
    def forward(self, t, meal_nutritions):
        """
        Args:
            t: (meal_records_length)
            meal_nutritions: (6)
        """
        t = t.unsqueeze(1) # (meal_records_length, 1)
        gamma_dist = (t**(self.k - 1) * torch.exp(-t/self.theta)) / (self.theta**self.k * torch.exp(torch.lgamma(self.k)))
        result = torch.sum(meal_nutritions * gamma_dist, dim=1) # (meal_records_length, 1)
        return result # (meal_records_length, 1)