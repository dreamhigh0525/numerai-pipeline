
import pandas as pd
from pathlib import Path
from numerapi import NumerAPI


def submit_predictions(napi: NumerAPI, model_name: str, file_path: str) -> str:
    model_id = napi.get_models()[model_name]
    print(f'model id: {model_id}')
    sub_id = napi.upload_predictions(file_path, model_id=model_id)
    return sub_id

def submit_example_predictions(napi: NumerAPI, dest: Path) -> str:
    live_example_preds = pd.read_parquet(f'{dest}/live_example_preds.parquet')
    live_example_preds["prediction"].to_csv('live_example_preds.csv')
    model_id = napi.get_models()['wf_example']
    sub_id = napi.upload_predictions('live_example_preds.csv', model_id=model_id)
    return sub_id