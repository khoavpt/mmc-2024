import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import kan

class KanNutritionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan_layer = kan.KAN(width=[1, 6], grid_range=[0, 40])

    def forward(self, t, meal_nutritions):
        """
        Args:
            t: (meal_records_length)
            meal_nutritions: (6)
        """
        nutritions_func = self.kan_layer(t.unsqueeze(1))
        result = torch.sum(nutritions_func * meal_nutritions, dim=1)
        return result

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
        plt.title('KAN Distributions for Each Nutrition', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()