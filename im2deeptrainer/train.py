import torch
from im2deeptrainer.model import IM2Deep
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar


def _data_to_dataloaders(data, batch_size, shuffle=True):
    tensors = {}
    for key in data.keys():
        tensors[key] = torch.tensor(
            data[key], dtype=torch.float32
        )  # TODO: check if dtype is correct, for y this is not specified in IM2DeepMulti

    dataset = torch.utils.data.TensorDataset(*[tensors[key] for key in data.keys()])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def _get_dataloaders(data, batch_size):
    train_dataloader = _data_to_dataloaders(data["train"], batch_size, shuffle=True)
    valid_dataloader = _data_to_dataloaders(data["valid"], batch_size, shuffle=False)
    test_dataloader = _data_to_dataloaders(data["test"], batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader

def _setup_callbacks(model_config):
    callbacks = [ModelSummary(), RichProgressBar()]
    if model_config["use_best_model"]:
        mcp = ModelCheckpoint(
            model_config["output_path"],
            save_best_only=True,
            filename=model_config["model_name"],
            monitor=model_config["monitor"],
            mode=model_config["mode"],
            save_last=False,
        )
        callbacks.append(mcp)

    return callbacks

def _setup_wandb_logger(model_config, model):
    wandb_config = model_config["wandb"]
    wandb_logger = WandbLogger(project=wandb_config["project_name"])
    wandb_logger.watch(model)
    return wandb_logger

def train_model(data, model_config):
    train_data, valid_data, test_data = _get_dataloaders(data, model_config["batch_size"])
    model = IM2Deep(model_config, criterion=model_config["criterion"])

    callbacks = _setup_callbacks(model_config)
    if model_config["wandb"]["enabled"]:
        wandb_logger = _setup_wandb_logger(model_config)

    trainer = pl.Trainer(
        devices=model_config["devices"],
        accelator="auto",
        max_epochs=model_config["n_epochs"],
        enable_progress_bar=True,
        callbacks=[callbacks],
        logger=wandb_logger if model_config["wandb"]["enabled"] else None,
    )

    trainer.fit(model, train_data, valid_data)

    # Load best model
    if model_config["use_best_model"]:
        model = IM2Deep.load_from_checkpoint(callbacks[-1].best_model_path, config=model_config, criterion=model_config["criterion"])

    return model


