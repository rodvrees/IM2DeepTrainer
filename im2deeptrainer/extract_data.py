from deeplcretrainer.cnn_functions import get_feat_df, get_feat_matrix
import os
import random
import pandas as pd
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)
random.seed(42)

MOL_FEATS = pd.read_csv(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "aa_mol_desc_feats.csv")
)


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


def _aa_chemical_features(features, mask=None):
    aa_features = features.iloc[:20]
    if mask:
        aa_features[aa_features.columns[[mask]]] = 0
    amino_acids = aa_features.set_index("Elements").T.to_dict("list")
    features_arrays = {
        aa: np.array(aa_features, dtype=np.float32) for aa, aa_features in amino_acids.items()
    }
    return features_arrays


def _mod_chemical_features(features, mask=None):
    mod_features = features.iloc[20:]
    if mask:
        mod_features[mod_features.columns[[mask]]] = 0
    mod_features = mod_features.set_index("Elements").T
    modified = mod_features.to_dict("list")
    dic = {}
    for key in modified:
        main_key, sub_key = key.split("#")
        dic.setdefault(main_key, {})[sub_key] = dict(zip(mod_features.index, modified[key]))
    return dic


def _empty_array():
    return np.zeros(
        (13, 60), dtype=np.float32
    )  # TODO: Adapt 13 this to the number of features, do not hardcode


def _string_to_tuple_list(input_string):
    parts = input_string.split("|")
    tuple_list = [(int(parts[i]), parts[i + 1]) for i in range(0, len(parts), 2)]
    return tuple_list


def encode_sequence_and_modification(sequence, modifications, modifications_dict, aa_to_feature):
    encoded = _empty_array()
    for i, aa in enumerate(sequence):
        encoded[:, i] = aa_to_feature[aa]

    try:
        modifications = _string_to_tuple_list(modifications)
    except:
        modifications = None
    if modifications:

        for position, mod in modifications:
            try:
                if position == 0:
                    encoded[:, position] = list(
                        modifications_dict[mod + "(N-T)"][sequence[position]].values()
                    )
                else:
                    encoded[:, position - 1] = list(
                        modifications_dict[mod][sequence[position - 1]].values()
                    )
            except KeyError:
                print(f"KeyError: {mod} {sequence[position-1]}")
                continue
    return encoded


def _get_mol_matrix(feat_df, features=MOL_FEATS):
    mol_feats = []

    aa_to_feature = _aa_chemical_features(features)
    mod_dict = _mod_chemical_features(features)

    for i in range(len(feat_df)):
        mol_feats.append(
            encode_sequence_and_modification(
                feat_df["seq"][i], feat_df["modifications"][i], mod_dict, aa_to_feature
            )
        )
    return np.array(mol_feats)


def _get_matrices(df, split_name, add_X_mol=True):
    # TODO: memory inneficient, fix
    if "tr" not in df.columns:
        if "CCS" not in df.columns:
            raise ValueError("CCS column not found in dataframe")
        else:
            df_renamed = df.rename(columns={"CCS": "tr"})

    feat_df = get_feat_df(df_renamed, predict_ccs=True)
    feat_df["charge"] = df_renamed["charge"]
    feat_df["seq"] = df_renamed["seq"]
    feat_df["modifications"] = df_renamed["modifications"]
    X, X_sum, X_global, X_hc, y = get_feat_matrix(feat_df)
    logger.debug("X shape: {}".format(X.shape))


    data = {
        f"X_{split_name}_AtomEnc": X,
        f"X_{split_name}_DiAminoAtomEnc": X_sum,
        f"X_{split_name}_GlobalFeatures": X_global,
        f"X_{split_name}_OneHot": X_hc,
        f"y_{split_name}": y,
    }

    if add_X_mol:
        X_mol = _get_mol_matrix(feat_df.reset_index(drop=True))
        data[f"X_{split_name}_MolEnc"] = X_mol
        logger.debug("X_mol shape {}:".format(str(X_mol.shape)))

    return data


def data_extraction(config):
    data = pd.read_csv(config["data_path"])
    ccs_df_train, ccs_df_test = _train_test_split(data, config["test_split"])
    ccs_df_train, ccs_df_valid = _train_test_split(ccs_df_train, config["val_split"])
    logger.debug(
        f"Train: {ccs_df_train.shape}, Valid: {ccs_df_valid.shape}, Test: {ccs_df_test.shape}"
    )
    train_data = _get_matrices(
        ccs_df_train, "train", add_X_mol=config["model_params"]["add_X_mol"]
    )
    valid_data = _get_matrices(
        ccs_df_valid, "valid", add_X_mol=config["model_params"]["add_X_mol"]
    )
    test_data = _get_matrices(ccs_df_test, "test", add_X_mol=config["model_params"]["add_X_mol"])

    if config["save_data_tensors"]:
        for split, data_dict in zip(
            ["train", "valid", "test"], [train_data, valid_data, test_data]
        ):
            for key, value in data_dict.items():
                pickle.dump(value, open(f"{config['output_path']}/{split}_{key}.pkl", "wb"))

    return {"train": train_data, "valid": valid_data, "test": test_data}, ccs_df_test
