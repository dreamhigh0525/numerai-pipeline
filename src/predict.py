#!/usr/bin/env python

from typing import List
import numpy as np
import pandas as pd
import pickle
from utils import load_model, neutralize, validation_metrics, ERA_COL


def predict_data(model_name: str, features: List[str], live_data: pd.DataFrame) -> pd.DataFrame:
    model = load_model(model_name)
    print(model)
    live_data = cast_features2int(live_data)
    print(live_data.shape)
    nans_per_col = live_data[live_data["data_type"] == "live"][features].isna().sum()
    # check for nans and fill nans
    if nans_per_col.any():
        total_rows = len(live_data[live_data["data_type"] == "live"])
        print(f"Number of nans per column this week: {nans_per_col[nans_per_col > 0]}")
        print(f"out of {total_rows} total rows")
        print(f"filling nans with 0.5")
        live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
    else:
        print("No nans in the features this week!")
    
    # double check the feature that the model expects vs what is available to prevent our
    # pipeline from failing if Numerai adds more data and we don't have time to retrain!
    model_expected_features = model.booster_.feature_name()
    if set(model_expected_features) != set(features):
        print(f"New features are available! Might want to retrain model {model_name}.")

    live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, model_expected_features])
    return live_data
    

def neutraize_data(filepath: str, model_name: str, live_data: pd.DataFrame) -> pd.DataFrame:
    with open(filepath, 'rb') as f:
        riskiest_features = pickle.load(f)

    live_data[f"preds_{model_name}_neutral_riskiest_50"] = neutralize(
        df=live_data,
        columns=[f"preds_{model_name}"],
        neutralizers=riskiest_features,
        proportion=1.0,
        normalize=True,
        era_col=ERA_COL
    )

    model_to_submit = f'preds_{model_name}_neutral_riskiest_50'
    live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
    return live_data


# cast float32 to int8
def cast_features2int(df: pd.DataFrame) -> pd.DataFrame:
  """Cast numerai quantile features to int for saving memory
  """
  cols = df.columns
  features = cols[cols.str.startswith('feature')].values.tolist()
  df_cast = (df[features].fillna(0.5) * 4).astype(np.int8)
  df_ = df[list(set(cols.values.tolist()) - set(features))]
  df = pd.concat([df_, df_cast], axis=1)
  df = df[cols]
  return df