import numpy as np
import wandb
import matplotlib.pyplot as plt
import logging
import torch
from scipy import stats
from im2deeptrainer.utils import MeanMAESorted, LowestMAESorted, MeanPearsonRSorted, MeanMRE

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


def _evaluate_predictions_multi(prediction1, prediction2, target1, target2):
    mean_mae = MeanMAESorted(target1, target2, prediction1, prediction2)
    lowest_mae = LowestMAESorted(target1, target2, prediction1, prediction2)
    mean_pearson_r = MeanPearsonRSorted(target1, target2, prediction1, prediction2)
    mean_mre = MeanMRE(target1, target2, prediction1, prediction2)

    return mean_mae, lowest_mae, mean_pearson_r, mean_mre


def _plot_predictions(test_df, predictions, mae, mean_pearson_r, mre, save_path=None, name=None):
    try:
        targets = test_df["CCS"].to_numpy()
    except KeyError:
        targets = test_df["tr"].to_numpy()
    charges = test_df["charge"].to_numpy()
    plt.scatter(targets, predictions, s=1, c=charges, cmap="viridis")
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], color="red")
    plt.xlabel("Observed CCS")
    plt.ylabel("Predicted CCS")
    plt.title(f"MAE: {mae:.2f}, Pearson R: {mean_pearson_r:.6f}, MRE: {mre:.5f}")
    plt.savefig(save_path + "predictions-{}.png".format(name))
    plt.close()


def _plot_predictions_multi(
    test_df,
    prediction1,
    prediction2,
    target1,
    target2,
    mean_mae,
    lowest_mae,
    mean_pearson_r,
    mean_mre,
    save_path=None,
    name=None,
):
    charges = test_df["charge"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(target1, prediction1, s=1, c=charges, cmap="viridis", label="Prediction 1")
    axes[0].plot([min(target1), max(target1)], [min(target1), max(target1)], color="red")
    axes[0].set_xlabel("Observed CCS")
    axes[0].set_ylabel("Predicted CCS")
    axes[1].scatter(target2, prediction2, s=1, c=charges, cmap="viridis", label="Prediction 2")
    axes[1].plot([min(target2), max(target2)], [min(target2), max(target2)], color="red")
    axes[1].set_xlabel("Observed CCS")
    axes[1].set_ylabel("Predicted CCS")
    fig.suptitle(
        f"Mean MAE: {mean_mae:.2f}, Lowest MAE: {lowest_mae:.2f}, Mean Pearson R: {mean_pearson_r:.6f}, Mean MRE: {mean_mre:.5f}"
    )
    plt.savefig(save_path + "predictions-{}-multi.png".format(name))
    plt.close()

    DeltaPred = abs(prediction1 - prediction2)
    DeltaCCS = abs(target1 - target2)

    fig, ax = plt.subplots()
    ax.scatter(DeltaCCS, DeltaPred, s=1, c=charges, cmap="viridis")
    ax.plot([min(DeltaCCS), max(DeltaCCS)], [min(DeltaCCS), max(DeltaCCS)], color="red")
    ax.set_xlabel("Difference between observed CCS")
    ax.set_ylabel("Difference between predicted CCS")
    ax.set_title("Delta CCS vs Delta Predicted CCS")
    plt.savefig(save_path + "predictions-{}-differences.png".format(name))
    plt.close()


def evaluate_and_plot(trainer, model, test_data, test_df, config):

    prediction_list = trainer.predict(model, test_data)
    predictions = np.concatenate(prediction_list)
    try:
        targets = test_df["CCS"].to_numpy()
    except KeyError:
        targets = test_df["tr"].to_numpy()

    if config["model_params"].get("multi-output", False) == False:

        test_mae, test_mean_pearson_r, test_mre = _evaluate_predictions(predictions, targets)

        if config["model_params"]["wandb"]["enabled"]:
            wandb.log(
                {"Test MAE": test_mae, "Test Pearson R": test_mean_pearson_r, "Test MRE": test_mre}
            )

        _plot_predictions(
            test_df,
            predictions,
            test_mae,
            test_mean_pearson_r,
            test_mre,
            config["output_path"],
            config["model_params"]["model_name"],
        )

        if config.get("output_predictions", False):
            test_df["predicted_CCS"] = predictions
            test_df.to_csv(
                config["output_path"]
                + "{}_predictions.csv".format(config["model_params"]["model_name"]),
                index=False,
            )
    else:
        predictions = np.sort(predictions, axis=1)
        targets = targets.reshape(-1, 1)
        targets = np.array([x[0] for x in targets])
        targets = np.sort(targets, axis=1)
        prediction1 = torch.tensor(predictions[:, 0])
        prediction2 = torch.tensor(predictions[:, 1])
        target1 = torch.tensor(targets[:, 0])
        target2 = torch.tensor(targets[:, 1])

        mean_mae, lowest_mae, mean_pearson_r, mean_mre = _evaluate_predictions_multi(
            prediction1, prediction2, target1, target2
        )

        if config["model_params"]["wandb"]["enabled"]:
            wandb.log(
                {
                    "Test Mean MAE": mean_mae,
                    "Test Lowest MAE": lowest_mae,
                    "Test Mean Pearson R": mean_pearson_r,
                    "Test Mean MRE": mean_mre,
                }
            )

        _plot_predictions_multi(
            test_df,
            prediction1,
            prediction2,
            target1,
            target2,
            mean_mae,
            lowest_mae,
            mean_pearson_r,
            mean_mre,
            config["output_path"],
            config["model_params"]["model_name"],
        )

        if config.get("output_predictions", False):
            test_df["observed_CCS_1"] = target1
            test_df["observed_CCS_2"] = target2
            test_df["predicted_CCS_1"] = prediction1
            test_df["predicted_CCS_2"] = prediction2
            test_df.to_csv(
                config["output_path"]
                + "{}_predictions.csv".format(config["model_params"]["model_name"]),
                index=False,
            )
