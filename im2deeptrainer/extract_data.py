from deeplcretrainer.cnn_functions import get_feat_df, get_feat_matrix
import random
import pandas as pd
import pickle
import logging

logger = logging.getLogger(__name__)
random.seed(42)

#TODO: Turn data into proforma format to enable Alireza features to be added

def _train_test_split(ccs_df, test_split=0.1):
    # Get all the indices of the dataframe
    indices = list(ccs_df.index)
    # Shuffle the indices
    random.shuffle(indices)
    # Split the indices in a training and testing set
    train_idx = indices[0 : int(len(indices) * (1 - test_split))]
    test_idx = indices[int(len(indices) * (1 - test_split)) :]

    logger.debug(int(len(indices) * (1 - test_split)))
    # Get the train and test indices and point to new variables
    ccs_df_train = ccs_df.loc[train_idx, :]
    ccs_df_test = ccs_df.loc[test_idx, :]
    return ccs_df_train, ccs_df_test


def _get_matrices(df, split_name):
    # TODO: memory inneficient, fix
    if 'tr' not in df.columns:
        if 'CCS' not in df.columns:
            raise ValueError("CCS column not found in dataframe")
        else:
            df_renamed = df.rename(columns={'CCS': 'tr'})

    feat_df = get_feat_df(df_renamed, predict_ccs=True)
    feat_df["charge"] = df_renamed["charge"]
    feat_df["seq"] = df_renamed["seq"]
    feat_df["modifications"] = df_renamed["modifications"]
    X, X_sum, X_global, X_hc, y = get_feat_matrix(feat_df)
    data = {
        f"X_{split_name}_AtomEnc": X,
        f"X_{split_name}_DiAminoAtomEnc": X_sum,
        f"X_{split_name}_GlobalFeatures": X_global,
        f"X_{split_name}_OneHot": X_hc,
        f"y_{split_name}": y,
    }
    return data


def data_extraction(config):
    data = pd.read_csv(config['data_path'])
    ccs_df_train, ccs_df_test = _train_test_split(data, config['test_split'])
    ccs_df_train, ccs_df_valid = _train_test_split(ccs_df_train, config['val_split'])
    logger.debug(f"Train: {ccs_df_train.shape}, Valid: {ccs_df_valid.shape}, Test: {ccs_df_test.shape}")
    train_data = _get_matrices(ccs_df_train, "train")
    valid_data = _get_matrices(ccs_df_valid, "valid")
    test_data = _get_matrices(ccs_df_test, "test")

    if config['save_data_tensors']:
        for split, data_dict in zip(['train', 'valid', 'test'], [train_data, valid_data, test_data]):
            for key, value in data_dict.items():
                pickle.dump(value, open(f"{config['output_path']}/{split}_{key}.pkl", "wb"))

    return {"train": train_data, "valid": valid_data, "test": test_data}, ccs_df_test
