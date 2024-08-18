import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import rootutils

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import src.models.nutrition_models as nm

class GIModel(pl.LightningModule):
    def __init__(self, X0=0, Gb=75.0, Ib=15.0, Gth=95.0, nutrition_model='gamma', optimizer_name='adam', lr=1e-3):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr

        # Initialize parameters 
        self.p1 = nn.Parameter(torch.tensor(0.05))
        self.p2 = nn.Parameter(torch.tensor(0.05))
        self.p3 = nn.Parameter(torch.tensor(0.05))
        self.p4 = nn.Parameter(torch.tensor(0.05))
        self.p5 = nn.Parameter(torch.tensor(Gth))
        self.p6 = nn.Parameter(torch.tensor(0.05))
        self.Gb = nn.Parameter(torch.tensor(Gb))
        self.Ib = nn.Parameter(torch.tensor(Ib))
        
        # Instantiate and register the GammaNutritionModel
        if nutrition_model == 'gamma':
            self.nutrition_model = nm.GammaNutritionModel()
        elif nutrition_model == 'exponential':
            self.nutrition_model = nm.ExponentialNutritionModel()
        elif nutrition_model == 'gaussian':
            self.nutrition_model = nm.GaussianNutritionModel()
        elif nutrition_model == 'mixed':
            self.nutrition_model = nm.MixedNutritionModel()
        elif nutrition_model == 'kan':
            self.nutrition_model = nm.KanNutritionModel()
        else:
            raise ValueError(f"Invalid nutrition model: {nutrition_model}")

        self.X0 = X0
        self.loss_fn = nn.MSELoss()
    
    def gi_dynamics(self, t, y):
        G, X, I = y
        nutrition = self.nutrition_model(t.unsqueeze(0), self.meal_nutritions)
        dGdt = - X * G + self.p1 * (self.Gb - G) + nutrition
        dXdt = - self.p2 * X + self.p3 * (I - self.Ib)
        dIdt = self.p4 * torch.relu(G - self.p5) - self.p6 * (I - self.Ib)
        return torch.stack([dGdt.squeeze(), dXdt, dIdt])

    def forward(self, t, gv, meal_nutritions):
        """
        Args:
            t: (meal_records_length)
            gv: (meal_records_length)
            meal_nutritions: (6) 
        """
        self.meal_nutritions = meal_nutritions
        y0 = torch.tensor([gv[0], self.X0, self.Ib], dtype=torch.float32)
        
        result = odeint(self.gi_dynamics, y0, t, atol=1e-5, rtol=1e-5)
        G_pred, _, _ = result[:, 0], result[:, 1], result[:, 2]
        return G_pred

    def training_step(self, batch, batch_idx):
        t, gv, meal_nutritions = batch
        t = t.squeeze() # (meal_records_length)
        gv = gv.squeeze() # (meal_records_length)
        meal_nutritions = meal_nutritions.squeeze() # (6, )

        # Predict the glucose value using the current parameters
        G_pred = self(t, gv, meal_nutritions)

        # Compute the loss
        loss = self.loss_fn(G_pred, gv.squeeze(0))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        t, gv, meal_nutritions = batch
        t = t.squeeze()
        gv = gv.squeeze()
        meal_nutritions = meal_nutritions.squeeze(1)

        # Predict the glucose value using the current parameters
        G_pred = self(t, gv, meal_nutritions)

        # Compute the loss
        loss = self.loss_fn(G_pred, gv.squeeze(0))
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer_name}")
        return optimizer

    def plot_prediction(self, glucose_dataset, userID, meal, filename=None):
        """
        Plot the glucose values and predicted values.

        Args:
            glucose_dataset: The dataset containing:
                - t: Time values (meal_records_length)
                - gv: Glucose values (meal_records_length)
                - meal_nutritions: Nutrition values (6)
            userID: The user ID
            meal: The meal type
            filename: The filename to save the plot (default: None)
        """
        sns.set(style="whitegrid")

        fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor='w')
        sns.lineplot(x=glucose_dataset[0].numpy(), y=glucose_dataset[1].numpy(), ax=ax, label='Glucose Value', color='b', alpha=0.7, lw=2, marker='o')
        sns.lineplot(x=glucose_dataset[0].numpy(), y=self(glucose_dataset[0], glucose_dataset[1], glucose_dataset[2]).detach().numpy(), ax=ax, label='Predicted Glucose Value', color='r', alpha=0.7, lw=2, marker='o')

        ax.set_xlabel('Time (days)', fontsize=14)
        ax.set_ylabel('Glucose Value', fontsize=14)
        ax.set_ylim(0, 200)
        ax.set_title(f'Glucose Value vs Predicted Glucose Value Over Time of {userID} after meal {meal}', fontsize=16)

        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)

        ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')

        legend = ax.legend(fontsize=12)
        legend.get_frame().set_alpha(0.5)

        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()