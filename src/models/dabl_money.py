'''
 Launches dabl tools over a set of CSVs (from datagouv usually)

Usage:
    dabl_dgf.py <i> [options]

Arguments:
    <i>                                A csv file or a folder with csv files
    --csv_detective_json FILE          A JSON file with the analysis of a csv_detective run over multiple CSVs [default: None:str]
    --num_cores=<n> CORES              Number of cores to use [default: 1:int]
'''
import argparse
from datetime import datetime
import os

from src.data.find_ml_candidates import find_interesting_mlearnable_datasets

today = datetime.today().strftime('%d_%m_%Y')

from pathlib import Path

import dabl
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm
import json

np.random.seed(42)


def get_files(input_folder, ext=".csv", n_sample=0):
    list_files = []
    for folder in os.listdir(input_folder):
        for file in os.listdir(input_folder / folder):
            # Open your file and run csv_detective
            file_path = input_folder / folder / file

            if file_path.suffix == ext:
                list_files.append(file_path)
    if n_sample > 0:
        list_files = sample(list_files, n_sample)
    return list_files


def load_csv_detective_cache(csv_detective_json: Path):
    """
    Try and load a JSON file that contains the analysis of a set of csv detectives
    :param csv_detective_json: Path of the analysis JSON file
    :return: A key:value dict csv_id:csv_detective_info
    """
    if csv_detective_json and csv_detective_json.exists():
        try:
            with open(csv_detective_json.as_posix()) as cache_path:
                csv_detective_cache = json.load(cache_path)
            return csv_detective_cache
        except:
            return {}
    else:
        return {}



def main(csv_file_path: Path, n_jobs: int, csv_detective_path: Path, output_path: Path):
    list_files = get_files(csv_file_path)
    # remove dabl analysis files
    list_files = [f for f in list_files if "dabl_" not in str(f)]

    output_folder = output_path / '_dabl'
    if not output_folder.exists():
        os.makedirs(output_folder)

    money_list, csv_detective_json = find_interesting_mlearnable_datasets(csv_detective_path)

    job_output = Parallel(n_jobs=n_jobs)(delayed(run)(csv_meta, list_files, output_folder)
                                         for csv_meta in tqdm(money_list.items()))

    clean_output = [j for j in job_output if j]
    nb_analyzed = len(clean_output)

    tqdm.write(f"We tried {len(list_files)} csv files, we could do at least one dabl model in {nb_analyzed}"
               f" files.")


def run(csv_metadata, list_files, output_folder):
    csv_id = csv_metadata[0]
    csv_metadata = csv_metadata[1]

    tqdm.write(f"\nTreating {csv_id} file")

    #Find the full path for the csv_id
    mask = [csv_id in str(f) for f in list_files]
    csv_file_path = list_files[mask.index(True)]

    dabl_analysis_path = (output_folder / (csv_id + '_dabl')).with_suffix('.csv')
    """
    if dabl_analysis_path.exists():
        tqdm.write(f"File {csv_id} already analyzed: {dabl_analysis_path} already exists")
        return dabl_analysis_path"""
    result_list = []

    if csv_metadata and len(csv_metadata) > 1:
        encoding = csv_metadata["encoding"]
        sep = csv_metadata["separator"]
    else:
        encoding = "latin-1"  # because why not
        sep = ";"
    csv_detective_columns = []
    if "columns" in csv_metadata:
        # keep columns that are not boolean
        csv_detective_columns = [k.strip('"') for k, v in csv_metadata['columns'].items() if "booleen" not in v]
    try:
        data: pd.DataFrame = pd.read_csv(csv_file_path.as_posix(), encoding=encoding, sep=sep, error_bad_lines=False)
        # remove csv_detective columns
        #data = data.drop(csv_detective_columns, axis=1)
        # TODO change this as now the columns are not in the same order

        data_clean, data_types = dabl.clean(data, return_types=True, verbose=3)
        # dabl.detect_types(data)
        money_variables = csv_metadata['columns']['money']
        for target_col in money_variables:
            try:
                data_clean_no_nan = data_clean[data_clean[target_col].notna()]
                if len(data_clean_no_nan) < 100:  # less than 100 examples is too few examples
                    continue
                print(f"Building models with target variable: {target_col}")
                sc = dabl.SimpleRegressor(random_state=42).fit(data_clean_no_nan, target_col=target_col)
                features_names = sc.est_.steps[0][1].get_feature_names()
                inner_dict = {"csv_id": csv_id, "task": "regression",
                              "algorithm": sc.current_best_.name,
                              "target_col": target_col,
                              "nb_features": len(features_names),
                              "features_names": "|".join(features_names),
                              "nb_classes": len(data[target_col].unique()),
                              "nb_lines": data_clean_no_nan.shape[0],
                              "date": today,
                              }

                inner_dict.update(sc.current_best_.to_dict())
                inner_dict.update({"avg_scores": np.mean(list(sc.current_best_.to_dict().values()))})
                result_list.append(inner_dict)
            except Exception as e:
                tqdm.write(f"Could not analyze file {csv_id} with target col {target_col}. Error {str(e)}")
    except Exception as e:
        tqdm.write(f"Could not analyze file {csv_id}. Error: {e}")
        return None
    if not result_list:
        return
    result_df = pd.DataFrame(result_list)
    with open(dabl_analysis_path, "w") as filo:
        result_df.to_csv(filo, header=True, index=False)
    return dabl_analysis_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder')
    parser.add_argument('--output_folder')
    parser.add_argument('--json_path')
    parser.add_argument('--num_cores',
                        default='1')

    args = parser.parse_args()

    raw_csv_path = Path(args.csv_folder)
    output_folder = Path(args.output_folder)
    csv_detective_path = Path(args.json_path)
    n_jobs = int(args.num_cores)

    main(raw_csv_path, n_jobs, csv_detective_path, output_folder)

