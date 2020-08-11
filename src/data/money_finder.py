import json
from pathlib import Path


analysis_json_path = Path('/home/robin/mlearnable-datasets-detective/data/output/2020-08-11_10-51-04.json')
csv_detective_json = json.load(open(analysis_json_path))

def find_with_money(csv_detective_json, min_nb=2, max_nb=40):
    money_list = {}
    for id, results in csv_detective_json.items():
        if len(results) < 2:  # this csv had some error
            continue
        money_columns = results['columns']['money']
        if len(money_columns) == 0:
            continue
        else:
            money_list[id] = results
    return money_list

if __name__ == '__main__':
    money_list = find_with_money(csv_detective_json, min_nb=2, max_nb=40)
    print(money_list)