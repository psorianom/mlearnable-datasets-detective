'''
Starts dabl over a given dataset resource id. Helps an expert determine if the analysis done by dabl makes sense and
thus if the dataset is truly machine-learnable
Checks different aspects such as correlations, features used,


Usage:
    find_ml_candidates.py <id> <dabl_output> <csv_detective_json> [options]

Arguments:
    <id>                    The id of the dataset--ressource to analyse
    <dabl_output>           The path of the analysis output file made by the dabl_dgf script
    <csv_detective_json>  The path to a csv detective analysis json
'''
import json

import sklearn
import pandas as pd
from argopt import argopt
import dabl

from src.models.utils import load_csv_detective_json



def main(resource_id, dabl_analysis_path, csv_detective_json):
    csv_detective_output = load_csv_detective_json(csv_detective_json=csv_detective_json) or {}
    dabl_output = json.load(csv_detective_json)


    pass


if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    resource_id = parser.id
    dabl_analysis_path = parser.output
    csv_detective_json = parser.csv_detective_json
    main(resource_id, dabl_analysis_path, csv_detective_json)

