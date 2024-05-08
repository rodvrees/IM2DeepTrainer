import logging
import click
from rich.logging import RichHandler
from rich.console import Console
import json
from pathlib import Path
from im2deeptrainer.exceptions import IM2DeepTrainerConfigError

# Relative imports
from im2deeptrainer.extract_data import data_extraction

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
    logging.basicConfig(
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=LOGGING_LEVELS[log_level],
        handlers=[
            RichHandler(
                rich_tracebacks=True, console=console, show_level=True, show_path=False
            )
        ],
    )

def _parse_config(config_path: Path):
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

    #TODO: parse config file
    config = _parse_config(config)
    # TODO: feature extraction
    train_data, valid_data, test_data = data_extraction(config)

    # TODO: model training
    raise NotImplementedError("Model training not implemented yet")

    # TODO: model evaluation and plotting
    raise NotImplementedError("Model evaluation and plotting not implemented yet")

    logger.info("Finished IM2DeepTrainer")

if __name__ == "__main__":
    main()

