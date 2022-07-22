#!/usr/bin/env python

import os
import json
import pandas as pd
from pathlib import Path
from numerapi import NumerAPI
from feature import FeatureSet
from utils import (
    save_model,
    load_model,
    neutralize,
    get_biggest_change_features,
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL,
    EXAMPLE_PREDS_COL
)

def download_data(napi: NumerAPI, dest: Path) -> None:
    print('download data')
    data_dir = dest.as_posix()
    #napi.download_dataset("v4/train.parquet")
    #napi.download_dataset("v4/validation.parquet")
    napi.download_dataset("v4/live.parquet", f"{data_dir}/live.parquet")
    napi.download_dataset("v4/live_example_preds.parquet", f"{data_dir}/live_example_preds.parquet")
    #napi.download_dataset("v4/validation_example_preds.parquet", "validation_example_preds.parquet")
    napi.download_dataset("v4/features.json", f"{data_dir}/features.json")


def read_metadata(feature_set: FeatureSet, dest: Path) -> pd.DataFrame:
    print(f'Reading {feature_set} data')
    # read the feature metadata and get a feature set (or all the features)
    data_dir = dest.as_posix()
    with open(f"{data_dir}/features.json", "r") as f:
        feature_metadata = json.load(f)
    
    if feature_set == FeatureSet.small:
        features = feature_metadata["feature_sets"]["small"]
    elif feature_set == FeatureSet.medium:
        features = feature_metadata["feature_sets"]["medium"]
    else:
        features = list(feature_metadata["feature_stats"].keys())
    
    # read in just those features along with era and target columns
    read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
    #print(read_columns)
    # note: sometimes when trying to read the downloaded data you get an error about invalid magic parquet bytes...
    # if so, delete the file and rerun the napi.download_dataset to fix the corrupted file
    #train_data = pd.read_parquet('v4/train.parquet', columns=read_columns)
    #print(train_data.shape)
    #val_data = pd.read_parquet('v4/validation.parquet', columns=read_columns)
    live_data = pd.read_parquet(f'{data_dir}/live.parquet', columns=read_columns)
    return live_data

if __name__ == '__main__':
    public_id = os.getenv('PUBLIC_ID')
    secret_key = os.getenv('SECRET_KEY')
    napi = NumerAPI(public_id=public_id, secret_key=secret_key)
    current_round = napi.get_current_round()
    print(f'current round: {current_round}')
    dest = Path(f'../data/v4/{current_round}')
    dest.mkdir(exist_ok=True, parents=False)
    download_data(napi, dest)
    live_data = read_metadata(FeatureSet.medium, dest)
    print(live_data.shape)
