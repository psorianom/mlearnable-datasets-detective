'''
 Launches dabl tools over a set of CSVs (from datagouv usually)

Usage:
    dabl_dgf.py <i> [options]

Arguments:
    <i>                                A csv file or a folder with csv files
    --csv_detective_json FILE          A JSON file with the analysis of a csv_detective run over multiple CSVs [default: None:str]
    --num_cores=<n> CORES              Number of cores to use [default: 1:int]
'''
import glob
from pathlib import Path

import dabl
import numpy as np
import pandas as pd
from argopt import argopt
from joblib import delayed, Parallel
from tqdm import tqdm
import json
from csv_detective.explore_csv import routine

np.random.seed(42)


def run(csv_file_path, csv_detective_cache):
    csv_metadata = get_csv_detective_metadata(csv_detective_cache=csv_detective_cache, csv_file_path=csv_file_path)
    if csv_metadata and len(csv_metadata) > 1:
        encoding = csv_metadata["encoding"]
        sep = csv_metadata["separator"]
    else:
        encoding = "latin-9"
        sep = ";"

    data: pd.DataFrame = pd.read_csv(csv_file_path.as_posix(), encoding=encoding, sep=sep)
    data_clean, data_types = dabl.clean(data, return_types=True, verbose=3)
    # dabl.detect_types(data)
    categorical_variables = data_types[data_types['categorical'] == True].index.values
    for cat_var in categorical_variables:
        print(f"Building models with target variable: {cat_var}")
        sc = dabl.SimpleClassifier(random_state=42).fit(data_clean, target_col=cat_var)
        print(sc.__dict__)
        dabl.explain(sc)
    pass


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


def get_csv_detective_metadata(csv_detective_cache: dict, csv_file_path: Path, num_rows=500):
    """
    Try and get the already computed meta-data of the csv_id passed, whether from a cached dict or calling
    the csv_detective routines
    :param csv_detective_cache: A key:value dict csv_id:csv_detective_info. Or an empty dict.
    :param csv_id: The id of the currently analysed csv file
    :return: The metadata of the csv file
    """
    csv_id = csv_file_path.stem
    if csv_detective_cache and csv_id in csv_detective_cache:
        return csv_detective_cache[csv_id]

    dict_result = routine(csv_file_path.as_posix(), num_rows=num_rows)
    csv_detective_cache[csv_id] = dict_result
    json.dump(csv_detective_cache, open("./data/csv_detective_analysis.json", "w"))
    return dict_result


def main(csv_path: Path, n_jobs: int, csv_detective_json: Path):
    csv_detective_cache = load_csv_detective_cache(csv_detective_json=csv_detective_json) or {}
    list_files = []

    if csv_path.exists():
        if csv_path.is_file():
            list_files = [csv_path]
        else:
            list_files = glob.glob(csv_path.as_posix() + "/**/*.csv", recursive=True)

    if n_jobs < 2:
        job_output = []
        for csv_file_path in tqdm(list_files):
            job_output.append(run(csv_file_path, csv_detective_cache))
    else:
        job_output = Parallel(n_jobs=n_jobs)(
            delayed(run)(csv_file_path, csv_detective_cache) for csv_file_path in tqdm(list_files))


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    csv_path = Path(parser.i)
    csv_detective_json = parser.csv_detective_json
    if parser.csv_detective_json:
        csv_detective_json = Path(parser.csv_detective_json)
    n_jobs = parser.num_cores

    main(csv_path, n_jobs, csv_detective_json)
