'''Analyzes a csv_detective analysis JSON file and finds datasets with continuous and categorical values.
    Those csvs with these kind of values are considered machine learnable

Usage:
    find_ml_candidates.py <i> [options]

Arguments:
    <i>                                The analysis JSON file generated by csv_detective
'''

import json

from .money_finder import find_with_money


def find_with_categorical(csv_detective_json, min_nb=2, max_nb=20):
    categorical_list = {}
    for id, results in csv_detective_json.items():
        if len(results) < 2:  # this csv had some error
            continue
        if "categorical" not in results:
            continue
        if min_nb > len(results["categorical"]) or len(results["categorical"]) > max_nb:
            continue
        # We discard those that are ids
        non_id_cols = []
        for column in results["categorical"]:
            if "columns_rb" in results:
                if column in results["columns_rb"]:
                    continue
            non_id_cols.append(column)
        if non_id_cols:
            results["categorical"] = non_id_cols
            categorical_list[id] = results

    return categorical_list


def find_with_continuous(csv_detective_json, min_nb=2, max_nb=40):
    continuous_list = {}
    for id, results in csv_detective_json.items():
        if len(results) < 2:  # this csv had some error
            continue
        if "continous" not in results:
            continue
        if min_nb > len(results["continous"]) or len(results["continous"]) > max_nb:
            continue
        # We discard those that are geo data
        for column in results["continous"]:
            if "columns_rb" not in results or column in results["columns_rb"]:
                continue
            continuous_list[id] = results

    return continuous_list


def find_with_categorical_and_continuous(categorical, continuous):
    intersect = set(categorical.keys()).intersection(continuous.keys())
    intersect_ds = {k: categorical[k] for k in intersect}
    return intersect_ds


def find_mlearnable_datasets(analysis_json_path):
    # 1. Load the file
    csv_detective_json = json.load(open(analysis_json_path))

    # 2. Find categorical AND continuous
    categorical = find_with_categorical(csv_detective_json)
    continuous = find_with_continuous(csv_detective_json)
    categorical_continuous = find_with_categorical_and_continuous(categorical, continuous)
    return categorical, continuous, categorical_continuous, csv_detective_json


def find_interesting_mlearnable_datasets(analysis_json_path):
    # 1. Load the file
    csv_detective_json = json.load(open(analysis_json_path))

    # 2. Find datasets talking about MONEY
    money_list = find_with_money(csv_detective_json)
    return money_list, csv_detective_json


if __name__ == '__main__':
    """
    parser = argopt(__doc__).parse_args()
    analysis_json_path = parser.i

    categorical, continuous, categorical_continuous, csv_detective_json = find_mlearnable_datasets(analysis_json_path)
    json.dump({"categorical": categorical, "continuous": continuous, "categorical_continuous": categorical_continuous},
              open("output/ml_candidates.json", "w"))
"""
