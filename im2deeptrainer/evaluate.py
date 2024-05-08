import numpy as np
import wandb
import matplotlib.pyplot as plt

def _mean_absolute_error(targets, predictions):
    return np.mean(np.abs(predictions - targets))

def _pearsonr(targets, predictions):
    return np.corrcoef(targets, predictions)[0, 1]

def _median_relative_error(targets, predictions):
    return np.median(np.abs(predictions - targets) / targets)

def _evaluate_predictions(predictions, targets):
    mae = _mean_absolute_error(targets, predictions)
    mean_pearson_r = _pearsonr(targets, predictions)[0]
    mre = _median_relative_error(targets, predictions)

    return mae, mean_pearson_r, mre

def _plot_predictions(targets, predictions, mae, mean_pearson_r, mre, save_path=None, name=None):
    plt.scatter(targets, predictions)
    plt.xlabel("True CCS")
    plt.ylabel("Predicted CCS")
    plt.title(f"MAE: {mae:.2f}, Pearson R: {mean_pearson_r:.2f}, MRE: {mre:.2f}")
    plt.savefig(save_path + "predictions-{}.png".format(name))


def evaluate_and_plot(trainer, model, test_data, test_df, config):
    predictions = trainer.predict(model, test_data)
    targets = test_df["CCS"].values

    test_mae, test_mean_pearson_r, test_mre = _evaluate_predictions(
        predictions, targets
    )

    if config['wandb']['enabled']:
        wandb.log({
            "test_mae": test_mae,
            "test_mean_pearson_r": test_mean_pearson_r,
            "test_mre": test_mre
        })

    _plot_predictions(
        targets, predictions, test_mae, test_mean_pearson_r, test_mre, config['output_path'], config['model_name']
    )




