import logging
import click
from rich.logging import RichHandler
from rich.console import Console

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


@click.command()
@click.option("--log-level", type=click.Choice(LOGGING_LEVELS.keys()), default="INFO")
@click.option("--config", type=click.Path(exists=True, dir_okay=False)) # TODO: add default in script as a variable, e.g. DEFAULT_CONFIG
def main(*args, **kwargs):
    _setup_logging(kwargs["log_level"])
    logger.info("Starting IM2DeepTrainer")

    # TODO: feature extraction
    raise NotImplementedError("Feature extraction not implemented yet")

    # TODO: model training
    raise NotImplementedError("Model training not implemented yet")

    # TODO: model evaluation and plotting
    raise NotImplementedError("Model evaluation and plotting not implemented yet")

    logger.info("Finished IM2DeepTrainer")

if __name__ == "__main__":
    main()

