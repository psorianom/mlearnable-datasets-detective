'''
Run dabl over a single CSV file or a folder with csv files

Usage:
    dabl_training.py <csv_path> [options]

Arguments:
    <csv_path>              CSV path or folder with CSVs path
    --optional_1 OPT1       Only convert those DOCs that are missing
    --cores=<n> CORES       Number of cores to use [default: 1:int]
'''
import logging
import os
from glob import glob
from pathlib import Path

from argopt import argopt
from joblib import Parallel, delayed
from tqdm import tqdm
import dabl
import pandas as pd


def run(doc_path):
    data = pd.read_csv(dabl.datasets.data_path(doc_path))
    detected_types = dabl.detect_types(data)
    data_clean = dabl.clean(data)[::10]
    return 1


def main(doc_files_path: Path, optional_1, n_jobs: int):
    if not os.path.isdir(doc_files_path) and os.path.isfile(doc_files_path):
        doc_paths = [doc_files_path]
    else:
        doc_paths = glob(doc_files_path + "/**/*.csv", recursive=True)
    if not doc_paths:
        raise Exception(f"Path {doc_paths} not found")

    if n_jobs < 2:
        job_output = []
    for doc_path in tqdm(doc_paths):
        tqdm.write(f"Converting file {doc_path}")
        job_output.append(run(doc_path))
    else:
        job_output = Parallel(n_jobs=n_jobs)(delayed(run)(doc_path) for doc_path in tqdm(doc_paths))

    logging.info(
        f"{sum(job_output)} DOC files were converted to TXT. {len(job_output) - sum(job_output)} files "
        f"had some error.")

    return doc_paths


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    doc_files_path = parser.csv_path
    optional_1 = parser.optional_1
    n_jobs = parser.cores
    main(doc_files_path=doc_files_path, optional_1=optional_1, n_jobs=n_jobs)
