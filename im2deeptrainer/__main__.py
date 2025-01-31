import logging
import click
from rich.logging import RichHandler
from rich.console import Console
import json
from pathlib import Path
from .exceptions import IM2DeepTrainerConfigError

# Relative imports
from .extract_data import data_extraction
from .train import train_model
from .evaluate import evaluate_and_plot

console = Console()
logger = logging.getLogger(__name__)

LOGGING_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _setup_logging(log_level):
    """Set up logging for IM2DeepTrainer"""
    logging.basicConfig(
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=LOGGING_LEVELS[log_level],
        handlers=[
            RichHandler(rich_tracebacks=True, console=console, show_level=True, show_path=True)
        ],
    )


def _setup_wandb(config):
    """Set up Weights and Biases logging"""
    wandb_config = config["model_params"]["wandb"]
    if wandb_config["enabled"]:
        import wandb

        wandb.init(
            project=wandb_config["project_name"],
            name=config["model_params"]["model_name"],
            save_code=False,
            config=config["model_params"],
        )
        config = wandb.config
    else:
        config = config["model_params"]
    return config


def _parse_config(config_path: Path):
    """Parse the config file"""
    if Path(config_path).suffix.lower() != ".json":
        raise IM2DeepTrainerConfigError("Config file must be a JSON file")

    with open(config_path, "r") as config_path:
        config = json.load(config_path)
        logger.debug(config)
    return config


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--log-level", type=click.Choice(LOGGING_LEVELS.keys()), default="INFO")
def main(config, *args, **kwargs):
    _setup_logging(kwargs["log_level"])
    logger.info("Starting IM2DeepTrainer")

    config = _parse_config(config)
    model_config = _setup_wandb(config)
    data, test_df = data_extraction(config)
    trainer, model, test_loader = train_model(
        data, model_config, output_path=config["output_path"]
    )
    evaluate_and_plot(trainer, model, test_loader, test_df, config)

    logger.info("Finished IM2DeepTrainer")


if __name__ == "__main__":
    import os

    # os.environ['WANDB_MODE']="offline"
    main()
