import tensorflow as tf
import numpy as np
import pandas as pd

from m4_evaluation_metrices import *
from glob import glob


def read_raw_data(file_path, no_headers = False):
    if no_headers:
        df = pd.read_csv(file_path, header=None)
    else:
        df = pd.read_csv(file_path)

    if 'V1' in df : del df['V1']
    
    return df.values

def evaluate_point_predictions(train_data, test_data, point_predcitions):
    errors = []

    for sample_x, sample_y, sample_prediction in zip(train_data, test_data, point_predcitions):
        sample_x = sample_x[~np.isnan(sample_x)]

        errors.append(m4_mase(sample_x, sample_y, sample_prediction))

    return np.array(errors).mean()

def evaluate_intervals_predictions(train_data, test_data, lower_predcitions, upper_predcitions):
    series_coverage = []
    msis_errors = []

    samples = zip(train_data, test_data, lower_predcitions, upper_predcitions)

    for sample_x, sample_y, sample_lower_prediction, sample_upper_prediction in samples :
        sample_x = sample_x[~np.isnan(sample_x)]

        series_coverage.append(coverage(sample_y, sample_lower_prediction, sample_upper_prediction))
        msis_errors.append(msis(sample_x, sample_y, sample_lower_prediction, sample_upper_prediction))

    acd_err = acd(np.array(series_coverage), test_data.shape[0]*test_data.shape[1])

    return acd_err, np.array(msis_errors).mean()


groups = ['new_models']

for group in groups:
    print(f'===============Evaluation results for {group} ===============')
    models = glob(f'results/{group}/*/')

    for model_path in models:
        for dataset in ['Test', 'Holdout']:

            print(f'==== For model {model_path}{dataset}')

            train_data = read_raw_data(f'Dataset/{dataset}/Hourly-train.csv')
            test_data = read_raw_data(f'Dataset/{dataset}/Hourly-test.csv')

            point_predictions = read_raw_data(f'{model_path}/{dataset}/point.csv', True)
            lower_predictions = read_raw_data(f'{model_path}/{dataset}/lower.csv', True)
            upper_predictions = read_raw_data(f'{model_path}/{dataset}/upper.csv', True)

            mase_err = evaluate_point_predictions(train_data, test_data, point_predictions)
            print(f'Point Prediction MASE {round(mase_err,3)}')

            smape_err = sMAPE(point_predictions, test_data)
            print(f'Point Prediction sMape {round(smape_err,3)}')

            acd_err, msis_err = evaluate_intervals_predictions(train_data, test_data, lower_predictions, upper_predictions)
            print(f'Interval Prediction ACD {round(acd_err,3)}')
            print(f'Interval Prediction MSIS {round(msis_err,3)} \n')

        