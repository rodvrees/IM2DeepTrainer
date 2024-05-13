import torch
import torch.nn as nn
import logging
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from im2deeptrainer.model import IM2Deep
from im2deeptrainer.model import LogLowestMAE

logger = logging.getLogger(__name__)

def _data_to_dataloaders(data, batch_size, shuffle=True):
    tensors = {}
    for key in data.keys():
        tensors[key] = torch.tensor(
            data[key], dtype=torch.float32
        )  # TODO: check if dtype is correct, for y this is not specified in IM2DeepMulti
        logger.debug(tensors[key].shape)

    dataset = torch.utils.data.TensorDataset(*[tensors[key] for key in data.keys()])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def _get_dataloaders(data, batch_size):
    train_dataloader = _data_to_dataloaders(data["train"], batch_size, shuffle=True)
    valid_dataloader = _data_to_dataloaders(data["valid"], batch_size, shuffle=False)
    test_dataloader = _data_to_dataloaders(data["test"], batch_size, shuffle=False)
    logger.debug(len(test_dataloader))
    return train_dataloader, valid_dataloader, test_dataloader

def _setup_callbacks(model_config, output_path):
    callbacks = [ModelSummary(), RichProgressBar(), LogLowestMAE()]
    if model_config["use_best_model"]:
        mcp = ModelCheckpoint(
            output_path + "/checkpoint",
            filename=model_config["model_name"],
            monitor=model_config["monitor"],
            mode=model_config["mode"],
            save_last=False,
        )
        callbacks.append(mcp)

    return callbacks

def _setup_wandb_logger(wandb_config, model):
    wandb_logger = WandbLogger(project=wandb_config["project_name"])
    wandb_logger.watch(model)
    return wandb_logger

def train_model(data, model_config, output_path):
    wandb_config = model_config["wandb"]
    train_data, valid_data, test_data = _get_dataloaders(data, model_config["batch_size"])
    model = IM2Deep(model_config, criterion=nn.L1Loss())
    logger.debug(model)

    callbacks = _setup_callbacks(model_config, output_path)
    if wandb_config["enabled"]:
        wandb_logger = _setup_wandb_logger(wandb_config, model)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[model_config["device"]],
        max_epochs=model_config["epochs"],
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=wandb_logger if wandb_config["enabled"] else None,
    )

    trainer.fit(model, train_data, valid_data)

    # Load best model
    if model_config["use_best_model"]:
        model = IM2Deep.load_from_checkpoint(callbacks[-1].best_model_path, config=model_config, criterion=nn.L1Loss())

    return trainer, model, test_data
    #TODO: save full model?


