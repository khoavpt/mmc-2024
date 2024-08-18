import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class GaussianNutritionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize parameters for each nutrition (calories, fat, carb, sugar, fiber, protein)
        self.t_meal = nn.Parameter(torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor([3.5, 3.5, 3.5, 3.5, 3.5, 3.5], dtype=torch.float32))
        
    def forward(self, t, meal_nutritions):
        """
        Args:
            t: (meal_records_length)
            meal_nutritions: (6)
        """
        t = t.unsqueeze(1) # (meal_records_length, 1)
        gaussian = (1 / (self.sigma * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-0.5 * ((t - self.t_meal) / self.sigma) ** 2)
        result = torch.sum(meal_nutritions * gaussian, dim=1) # (meal_records_length, 1)
        return result # (meal_records_length, 1)
    
    def plot_nutrition_distributions(self, filename=None):
        t = torch.linspace(0, 40, steps=100)  # Generate 100 points from 0 to 40
        meal_nutritions = torch.eye(6)  # Identity matrix to isolate each nutrition
        nutrition_labels = ['Calories', 'Fat', 'Carb', 'Sugar', 'Fiber', 'Protein']

        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))

        for i in range(6):
            nutrition_dist = self.forward(t, meal_nutritions[i])
            sns.lineplot(x=t.numpy(), y=nutrition_dist.detach().numpy(), label=nutrition_labels[i], lw=2)

        plt.xlabel('Time (t)', fontsize=14)
        plt.ylabel('Distribution', fontsize=14)
        plt.title('Gaussian Distributions for Each Nutrition', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()
