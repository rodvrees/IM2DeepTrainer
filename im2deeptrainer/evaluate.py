import numpy as np
import wandb
import matplotlib.pyplot as plt
import logging
import torch

logger = logging.getLogger(__name__)

def _mean_absolute_error(targets, predictions):
    logger.debug('Here2')
    differences = np.abs(predictions - targets)
    mae = np.mean(differences)
    del differences
    return mae

def _pearsonr(targets, predictions):
    logger.debug('Here3')
    return np.corrcoef(targets, predictions)[0]

def _median_relative_error(targets, predictions):
    logger.debug('Here4')
    return np.median(np.abs(predictions - targets) / targets)

def _evaluate_predictions(predictions, targets):
    mae = _mean_absolute_error(targets, predictions)
    mean_pearson_r = _pearsonr(targets, predictions)[0]
    mre = _median_relative_error(targets, predictions)

    return mae, mean_pearson_r, mre

def _plot_predictions(targets, predictions, mae, mean_pearson_r, mre, save_path=None, name=None):
    logger.debug('Here5')
    plt.scatter(targets, predictions, s=1)
    plt.xlabel("True CCS")
    plt.ylabel("Predicted CCS")
    plt.title(f"MAE: {mae:.2f}, Pearson R: {mean_pearson_r:.6f}, MRE: {mre:.4f}")
    plt.savefig(save_path + "predictions-{}.png".format(name))


def evaluate_and_plot(trainer, model, test_data, test_df, config):
    prediction_list = trainer.predict(model, test_data)
    predictions = np.concatenate(prediction_list)
    targets = test_df["CCS"].to_numpy()
    test_mae, test_mean_pearson_r, test_mre = _evaluate_predictions(
        predictions, targets
    )

    if config['model_params']['wandb']['enabled']:
        wandb.log({
            "test_mae": test_mae,
            "test_mean_pearson_r": test_mean_pearson_r,
            "test_mre": test_mre
        })

    _plot_predictions(
        targets, predictions, test_mae, test_mean_pearson_r, test_mre, config['output_path'], config['model_params']['model_name']
    )




