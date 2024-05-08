from deeplcretrainer.cnn_functions import get_feat_df, get_feat_matrix
import random
import pandas as pd
import pickle

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

    # Get the train and test indices and point to new variables
    ccs_df_train = ccs_df.loc[train_idx, :]
    ccs_df_test = ccs_df.loc[test_idx, :]
    return ccs_df_train, ccs_df_test


def _get_matrices(df, split_name):
    df = get_feat_df(df, predict_ccs=True)
    df["charge"] = df["charge"]
    df["seq"] = df["seq"]
    df["modifications"] = df["modifications"]
    X, X_sum, X_global, X_hc, y = get_feat_matrix(df)
    data = {
        f"X_{split_name}_AtomEnc": X,
        f"X_{split_name}_DiAminoAtomEnc": X_sum,
        f"X_{split_name}_GlobalFeatures": X_global,
        f"X_{split_name}_OneHot": X_hc,
        f"y_{split_name}": y,
    }
    return data


def data_extraction(data_path, config):
    data = pd.read_csv(data_path)
    ccs_df_train, ccs_df_test = _train_test_split(data)
    ccs_df_train, ccs_df_valid = _train_test_split(ccs_df_test)
    train_data = _get_matrices(ccs_df_train, "train")
    valid_data = _get_matrices(ccs_df_valid, "valid")
    test_data = _get_matrices(ccs_df_test, "test")

    if config['save_data_tensors']:
        for split, data_dict in zip(['train', 'valid', 'test'], [train_data, valid_data, test_data]):
            for key, value in data_dict.items():
                pickle.dump(value, open(f"{config['data_path']}/{split}_{key}.pkl", "wb"))

    return train_data, valid_data, test_data
