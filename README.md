# mlearnable-datasets-detective
Try and discover which datasets from data.gouv.fr are useful for a supervised learning task.

## Note for using dabl_money
you might get an error using the pip version of dabl see 
https://github.com/dabl/dabl/pull/216 and modify venv/lib/python3.6/site-packages/dabl/preprocessing.py

## How to run
python -m src.models.dabl_money --csv_folder /home/robin/mlearnable-datasets-detective/data/csv --json_path /home/robin/mlearnable-datasets-detective/data/output/2020-08-12_09-32-40.json --output_folder /home/robin/mlearnable-datasets-detective/data/output
