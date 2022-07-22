

from numerapi import NumerAPI

def submit_predictions(napi: NumerAPI, model_name: str, file_path: str):
    model_id = napi.get_models()[model_name]
    napi.upload_predictions(file_path, model_id=model_id)