from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from core.utils.gauss_rank_scaler import GaussRankScaler

from core.bll.preprocessing import Preprocessor
from core.bll.ordered_data_reader import OrderedDataReader
from core.bll.features import Features

def scale_group(df):
    f = Features()
    force = f.get_vpp_signal(df['force'].values)
    cole_params = np.array([f.get_cole_cole_params(df[col].values) for col in input_columns])
    cole_params = np.moveaxis(cole_params, 1, 0).reshape(f.feature_length, -1)
    return cole_params, force

def get_data(df):
    features_list, force_list = [], []
    for index, df_ in df.groupby(['subject', 'signal_num']):
        features, force = scale_group(df_)
        features_list.append(features)
        force_list.append(force)

    features_list = np.array(features_list).reshape(-1, 24)
    force_list = np.array(force_list).reshape(-1)
    return features_list, force_list

if __name__ == '__main__':
    input_columns = [f'channel_{c}' for c in range(8)]
    train_annotations_path = Path(__file__).joinpath('..', 'files', 'train_annotations.csv').resolve()
    val_annotations_path = Path(__file__).joinpath('..', 'files', 'val_annotations.csv').resolve()
    train_df = pd.read_csv(train_annotations_path)
    val_df = pd.read_csv(val_annotations_path)
    #train_df = train_df.groupby(['subject', 'signal_num']).apply(scale_group)
    train_features, train_labels = get_data(train_df)
    val_features, val_labels = get_data(val_df)
    reg = XGBRegressor().fit(train_features, train_labels)
    s = reg.score(val_features, val_labels)
    print(s)