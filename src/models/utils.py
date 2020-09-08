from pathlib import Path

import dabl
from tqdm import tqdm
from csv_detective.explore_csv import routine

def get_csv_detective_metadata(csv_detective_cache: dict, csv_file_path: Path, num_rows=5000):
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


def preprocess_data(df, sample=20000):
    data = df.sample(n=min(sample, len(df)), random_state=42)
    data_clean, data_types = dabl.clean(data, return_types=True, verbose=3)
    return data_clean, data_types


def load_dgf_df(csv_file_path, csv_detective_json):
    tqdm.write(f"\nTreating {csv_file_path} file")
    csv_file_path = Path(csv_file_path)
    csv_id = csv_file_path.stem
    dabl_analysis_path = Path(f"{csv_file_path.as_posix()[:-4]}_dabl.csv")
    if dabl_analysis_path.exists():
        tqdm.write(f"File {csv_id} already analyzed: {dabl_analysis_path} already exists")
        return dabl_analysis_path
    result_list = []
    csv_metadata = get_csv_detective_metadata(csv_detective_cache=csv_detective_json, csv_file_path=csv_file_path)
    if csv_metadata and len(csv_metadata) > 1:
        encoding = csv_metadata["encoding"]
        sep = csv_metadata["separator"]
    else:
        encoding = "latin-1"  # bcause why not
        sep = ";"
    pass


def load_csv_detective_json(csv_detective_json: Path):
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
