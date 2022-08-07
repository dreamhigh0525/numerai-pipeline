#!/usr/bin/env python

import argparse
import os
import datetime
from pathlib import Path
from numerapi import NumerAPI
from feature import FeatureSet
from preprocess import download_data, read_metadata
from predict import neutraize_data, predict_data
from submit import submit_predictions, submit_example_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Numerai submit')
    parser.add_argument(
        '--model', '-m', required=True, type=str,
        help='model name[small/medium/all/example]'
    )
    parser.add_argument(
        '--download', '-d', action='store_true',
        help='download numerai data'
    )
    parser.add_argument(
        '--submit', '-s', action='store_true',
        help='submit prediction'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    model_name = f'wf_{args.model}'
    print(model_name)
    public_id = os.getenv('PUBLIC_ID')
    secret_key = os.getenv('SECRET_KEY')
    napi = NumerAPI(public_id=public_id, secret_key=secret_key)
    current_round = napi.get_current_round()
    print(f'current round: {current_round}')
    dest = Path(f'../data/v4/{current_round}')
    if args.download:
        dest.mkdir(exist_ok=True, parents=False)
        download_data(napi, dest)
    
    if args.model == 'example':
        submit_example_predictions(napi, dest)
    else:
        live_data, features = read_metadata(args.model, dest)
        print(live_data.shape)
        live_data = predict_data(model_name, features, live_data)
        neutraize_filepath = f'../models/riskiest_features_{args.model}.pkl'
        live_data = neutraize_data(neutraize_filepath, model_name, live_data)
        print(live_data.head())
        today = datetime.date.today().strftime('%m%d')
        pred_filename = f"../pred/live_preds_{model_name}_{current_round}_{today}.csv"
        print(f'write {pred_filename}')
        live_data["prediction"].to_csv(pred_filename)
        if args.submit:
            sub_id = submit_predictions(napi, model_name, pred_filename)
            print(f'submit success: {sub_id}')
