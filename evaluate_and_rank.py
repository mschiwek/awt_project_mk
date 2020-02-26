import numpy as np
import pandas as pd
import json
import io
import prettytable

from m4_evaluation_metrices import *
from glob import glob


def read_raw_data(file_path, no_headers=False):
    if no_headers:
        df = pd.read_csv(file_path, header=None)
    else:
        df = pd.read_csv(file_path)

    if 'V1' in df: del df['V1']

    return df.values


def read_config(file_path):
    # json config for recreation purposes
    with open(file_path) as fp:
        config = json.load(fp)
    # append dropout for models without dropout
    if not "dropout" in config:
        config["dropout"] = None
        with open(file_path, 'w') as fp:
            json.dump(config, fp)

    return config


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

    for sample_x, sample_y, sample_lower_prediction, sample_upper_prediction in samples:
        sample_x = sample_x[~np.isnan(sample_x)]

        series_coverage.append(coverage(sample_y, sample_lower_prediction, sample_upper_prediction))
        msis_errors.append(msis(sample_x, sample_y, sample_lower_prediction, sample_upper_prediction))

    acd_err = acd(np.array(series_coverage), test_data.shape[0] * test_data.shape[1])

    return acd_err, np.array(msis_errors).mean()


groups = ['new_models']

for group in groups:
    print(f'===============Evaluation results for {group} ===============')
    models = glob(f'results/{group}/*/')
    res_test = []
    res_holdout = []

    for model_path in models:
        for dataset in ['Test', 'Holdout']:
            print(f'==== For model {model_path}{dataset}')

            train_data = read_raw_data(f'Dataset/{dataset}/Hourly-train.csv')
            test_data = read_raw_data(f'Dataset/{dataset}/Hourly-test.csv')

            point_predictions = read_raw_data(f'{model_path}/{dataset}/point.csv', True)
            lower_predictions = read_raw_data(f'{model_path}/{dataset}/lower.csv', True)
            upper_predictions = read_raw_data(f'{model_path}/{dataset}/upper.csv', True)

            config = read_config(f'{model_path}config.json')

            mase_err = evaluate_point_predictions(train_data, test_data, point_predictions)
            print(f'Point Prediction MASE {round(mase_err, 3)}')

            smape_err = sMAPE(point_predictions, test_data)
            print(f'Point Prediction sMape {round(smape_err, 3)}')

            acd_err, msis_err = evaluate_intervals_predictions(train_data, test_data, lower_predictions,
                                                               upper_predictions)
            print(f'Interval Prediction ACD {round(acd_err, 3)}')
            print(f'Interval Prediction MSIS {round(msis_err, 3)} \n')

            if dataset == "Holdout":
                res = res_holdout
            else:
                res = res_test

            res.append({"dataset": dataset,
                        "look_back": config["look_back"],
                        "hop": config["hop"],
                        "feature_dim": config["feature_dim"],
                        "msis": round(msis_err, 3),
                        "acd": round(acd_err, 3),
                        "smape": round(smape_err, 3),
                        "mase": round(mase_err, 3),
                        "dropout": config["dropout"],
                        "lube_loss": config["lube_loss"],
                        "point_loss": config["point_loss"],
                        "hidden_units": config["hidden_units"],
                        "hidden_layers": config["hidden_layers"]
                        })
    if len(res_holdout) > 0:
        for error in ["msis", "smape"]:
            for res in [res_test, res_holdout]:
                sorted_res = sorted(res, key=lambda i: i[error])
                data = []
                # choose correct loss_metric
                if error == "msis":
                    error1 = "acd"
                    error_metric = "lube_loss"
                else:
                    error1 = "mase"
                    error_metric = "point_loss"

                i = 0
                # print 5 best results
                for elem in sorted_res:
                    data.append(
                        [elem[error], elem[error1], elem["look_back"], elem["hop"], elem["feature_dim"],
                         elem["dropout"],
                         elem[error_metric], elem["hidden_units"], elem["hidden_layers"]])

                df = pd.DataFrame(data,
                                  columns=[error, error1, "Lookback", "Hop", "Features", "Dropout", "Loss",
                                           "Hidden Units",
                                           "Hidden Layers"])

                # print prettytable
                output = io.StringIO()
                df.to_csv(output)
                output.seek(0)
                print(sorted_res[0]["dataset"])
                print(prettytable.from_csv(output))
                print("\n\n")

                # export to excel file
                filename = error + "_" + sorted_res[0]["dataset"] + "_res.xlsx"
                print(filename)
                print(df)
                df.to_excel(filename, "Sheet1")
