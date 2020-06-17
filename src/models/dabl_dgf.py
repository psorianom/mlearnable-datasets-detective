'''
 Launches dabl tools over a set of CSVs (from datagouv usually)

Usage:
    dabl_dgf.py <i> [options]

Arguments:
    <i>                                A csv file or a folder with csv files
    --csv_detective_json FILE          A JSON file with the analysis of a csv_detective run over multiple CSVs [default: None:str]
    --num_cores=<n> CORES              Number of cores to use [default: 1:int]
'''
from datetime import datetime

today = datetime.today().strftime('%d_%m_%Y')
import glob
from collections import defaultdict
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
    tqdm.write(f"Treating {csv_file_path} file")
    csv_file_path = Path(csv_file_path)
    csv_id = csv_file_path.stem
    results_list = []
    csv_metadata = get_csv_detective_metadata(csv_detective_cache=csv_detective_cache, csv_file_path=csv_file_path)
    if csv_metadata and len(csv_metadata) > 1:
        encoding = csv_metadata["encoding"]
        sep = csv_metadata["separator"]
    else:
        encoding = "latin-1"  # bcause why not
        sep = ";"
    csv_detective_columns = []
    if "columns" in csv_metadata:
        csv_detective_columns = [k.strip('"') for k, v in csv_metadata['columns'].items() if "booleen" not in v]

    data: pd.DataFrame = pd.read_csv(csv_file_path.as_posix(), encoding=encoding, sep=sep)
    # remove csv_detective columns
    data = data.drop(csv_detective_columns, axis=1)
    data_clean, data_types = dabl.clean(data, return_types=True, verbose=3)
    # dabl.detect_types(data)
    categorical_variables = data_types[data_types['categorical']].index.values
    for target_col in categorical_variables:
        try:
            if data_clean[target_col].isna().sum():
                continue
            print(f"Building models with target variable: {target_col}")
            sc = dabl.SimpleClassifier(random_state=42).fit(data, target_col=target_col)
            inner_dict = {"csv_id": csv_id, "task": "classification",
                          "algorithm": sc.current_best_.name,
                          "target_col": target_col,
                          "features_names": "|".join(sc.feature_names_),
                          "date": today,
                          "nb_classes": len(data[target_col].unique())
                          }
            inner_dict.update(sc.current_best_.to_dict())
            inner_dict.update({"avg_scores": np.mean(list(sc.current_best_.to_dict().values()))})
            results_list.append(inner_dict)
        except Exception as e:
            tqdm.write(f"Could not analyse file {csv_id} with target col {target_col}. Error {str(e)}")
            results_list.append(None)

        # dabl.explain(sc)
    return results_list


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
    csv_file_path = Path(csv_file_path)
    csv_id = csv_file_path.stem
    if csv_detective_cache and csv_id in csv_detective_cache:
        return csv_detective_cache[csv_id]
    try:
        dict_result = routine(csv_file_path.as_posix(), num_rows=num_rows)
    except:
        return {}
    csv_detective_cache[csv_id] = dict_result
    json.dump(csv_detective_cache, open("./data/csv_detective_analysis.json", "w"), indent=4)
    return dict_result


def main(csv_file_path: Path, n_jobs: int, csv_detective_json: Path):
    csv_detective_cache = load_csv_detective_cache(csv_detective_json=csv_detective_json) or {}
    list_files = []

    if csv_file_path.exists():
        if csv_file_path.is_file():
            list_files = [csv_file_path]
        else:
            list_files = glob.glob(csv_file_path.as_posix() + "/**/*.csv", recursive=True)

    if n_jobs < 2:
        job_output = []
        dabl_analyzed_count = 0
        csv_analyzed_total = 0
        for csv_file_path in tqdm(list_files):
            csv_analyzed_total += 1
            dabl_result = run(csv_file_path, csv_detective_cache)
            if all(v is None for v in dabl_result):  # all contents are None
                continue
            dabl_analyzed_count += 1
            job_output.extend(dabl_result)

    else:
        job_output = Parallel(n_jobs=n_jobs)(
            delayed(run)(csv_file_path, csv_detective_cache) for csv_file_path in tqdm(list_files))

    json.dump(job_output, open("./data/dabl_analysis.json", "w"), indent=4)
    tqdm.write(f"We tried {csv_analyzed_total} csv files, we could do at least one dabl model in {dabl_analyzed_count}"
               f" files.")


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    csv_path = Path(parser.i)
    csv_detective_json = parser.csv_detective_json
    if parser.csv_detective_json:
        csv_detective_json = Path(parser.csv_detective_json)
    n_jobs = parser.num_cores

    main(csv_path, n_jobs, csv_detective_json)
