import numpy as np
import wandb
import matplotlib.pyplot as plt
import logging
import torch
from scipy import stats

logger = logging.getLogger(__name__)

def _mean_absolute_error(targets, predictions):
    differences = np.abs(predictions - targets)
    mae = np.mean(differences)
    del differences
    return mae

def _pearsonr(targets, predictions):
    return stats.pearsonr(targets, predictions).statistic

def _median_relative_error(targets, predictions):
    return np.median(np.abs(predictions - targets) / targets)

def _evaluate_predictions(predictions, targets):
    mae = _mean_absolute_error(targets, predictions)
    mean_pearson_r = _pearsonr(targets, predictions)
    mre = _median_relative_error(targets, predictions)

    return mae, mean_pearson_r, mre

def _plot_predictions(test_df, predictions, mae, mean_pearson_r, mre, save_path=None, name=None):
    try:
        targets = test_df["CCS"].to_numpy()
    except KeyError:
        targets = test_df["tr"].to_numpy()
    charges = test_df["charge"].to_numpy()
    plt.scatter(targets, predictions, s=1, c=charges, cmap="viridis")
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], color="red")
    plt.xlabel("True CCS")
    plt.ylabel("Predicted CCS")
    plt.title(f"MAE: {mae:.2f}, Pearson R: {mean_pearson_r:.6f}, MRE: {mre:.5f}")
    plt.savefig(save_path + "predictions-{}.png".format(name))


def evaluate_and_plot(trainer, model, test_data, test_df, config):
    prediction_list = trainer.predict(model, test_data)
    predictions = np.concatenate(prediction_list)
    try:
        targets = test_df["CCS"].to_numpy()
    except KeyError:
        targets = test_df["tr"].to_numpy()
    test_mae, test_mean_pearson_r, test_mre = _evaluate_predictions(
        predictions, targets
    )

    if config['model_params']['wandb']['enabled']:
        wandb.log({
            "Test MAE": test_mae,
            "Test Pearson R": test_mean_pearson_r,
            "Test MRE": test_mre
        })

    _plot_predictions(
        test_df, predictions, test_mae, test_mean_pearson_r, test_mre, config['output_path'], config['model_params']['model_name']
    )




