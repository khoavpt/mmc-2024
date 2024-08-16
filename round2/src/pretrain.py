import hydra
from omegaconf import DictConfig
import rootutils
import logging
import pandas as pd
import pytorch_lightning as pl

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
CONFIGPATH = str(ROOTPATH / "configs")

import src.data.dataset as dataset
import src.models.glucose_insulin as gi
import src.utils.utils as utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path=CONFIGPATH, config_name="config", version_base=None)
def main(cfg:DictConfig):
    # Load data
    pretrain_dataset: dataset.GlucoseRecordings = hydra.utils.instantiate(cfg.dataset)
    pretrain_dataloader = pretrain_dataset.get_data_loader()
    
    # Load model
    model: gi.GIModel = hydra.utils.instantiate(cfg.model)

    # Train the model
    loss_logger = utils.LossLogger()
    trainer = pl.Trainer(max_epochs=20, callbacks=[loss_logger])
    trainer.fit(model, pretrain_dataloader)

    # Plot and save the losses
    loss_logger.plot_losses(ROOTPATH / "figures" / f"{cfg.model.nutrition_model}_pretrain_loss.png")
    logger.info(f"Train losses: {str(loss_logger.train_losses)} ")

    # Save the model
    utils.save_model(model, ROOTPATH / "pretrained_models" / f"{cfg.model.nutrition_model}_model.pth")

if __name__ == "__main__":
    main()