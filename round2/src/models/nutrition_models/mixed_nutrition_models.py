import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class MixedNutritionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize parameters for Gamma distribution (calories, fat)
        self.k_gamma = nn.Parameter(torch.tensor([8.0, 8.0], dtype=torch.float32))
        self.theta_gamma = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        
        # Initialize parameters for Gaussian distribution (carb, sugar, protein)
        self.t_meal_gaussian = nn.Parameter(torch.tensor([20.0, 10.0, 20.0], dtype=torch.float32))
        self.sigma_gaussian = nn.Parameter(torch.tensor([3.0, 1.0, 3.0], dtype=torch.float32))
    
    def forward(self, t, meal_nutritions):
        """
        Args:
            t: (meal_records_length)
            meal_nutritions: (6)
        """
        t = t.unsqueeze(1) # (meal_records_length, 1)
        
        # Gamma distribution for calories and fat
        gamma_dist_calories = (t**(self.k_gamma[0] - 1) * torch.exp(-t/self.theta_gamma[0])) / (self.theta_gamma[0]**self.k_gamma[0] * torch.exp(torch.lgamma(self.k_gamma[0])))
        gamma_dist_fat = (t**(self.k_gamma[1] - 1) * torch.exp(-t/self.theta_gamma[1])) / (self.theta_gamma[1]**self.k_gamma[1] * torch.exp(torch.lgamma(self.k_gamma[1])))
        
        # Gaussian distribution for carb, sugar, protein
        gaussian_dist_carb = (1 / (self.sigma_gaussian[0] * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-0.5 * ((t - self.t_meal_gaussian[0]) / self.sigma_gaussian[0]) ** 2)
        gaussian_dist_sugar = (1 / (self.sigma_gaussian[1] * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-0.5 * ((t - self.t_meal_gaussian[1]) / self.sigma_gaussian[1]) ** 2)
        gaussian_dist_protein = (1 / (self.sigma_gaussian[2] * torch.sqrt(torch.tensor(2 * torch.pi)))) * torch.exp(-0.5 * ((t - self.t_meal_gaussian[2]) / self.sigma_gaussian[2]) ** 2)
        
        # Zero distribution for fiber
        zero_dist_fiber = torch.zeros_like(t)
        
        # Combine distributions in the correct order
        combined_dist = torch.cat([gamma_dist_calories, gamma_dist_fat, gaussian_dist_carb, gaussian_dist_sugar, zero_dist_fiber, gaussian_dist_protein], dim=1)
        
        result = torch.sum(meal_nutritions * combined_dist, dim=1) # (meal_records_length, 1)
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
        plt.title('Distributions for Each Nutrition', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()