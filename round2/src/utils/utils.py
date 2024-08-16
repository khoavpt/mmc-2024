import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

class LossLogger(pl.Callback):
    def __init__(self):
        self.train_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss')
        self.train_losses.append(train_loss.item())
    
    def plot_losses(self, filename=None):
        if self.train_losses:
            sns.set(style="whitegrid")
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

            # Plot training
            sns.lineplot(x=range(len(self.train_losses)), y=self.train_losses, ax=ax1, label='Train Loss', marker='o')
            ax1.set_xlabel('Epoch', fontsize=14)
            ax1.set_ylabel('Loss', fontsize=14)
            ax1.legend(fontsize=12)
            ax1.set_title('Training Loss (MSE) Over Epochs', fontsize=16)
            plt.tight_layout()
            if filename:
                plt.savefig(filename)
            else:
                plt.show()
        else:
            print("No data to plot.")

def save_model(model, path):
    """
    Save the model to the specified path.

    Args:
        model: The model to be saved.
        path: The path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, *args, **kwargs):
    """
    Load the model from the specified path.

    Args:
        model_class: The class of the model to be loaded.
        path: The path from where the model will be loaded.
        *args: Additional arguments to initialize the model.
        **kwargs: Additional keyword arguments to initialize the model.

    Returns:
        The loaded model.
    """
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model