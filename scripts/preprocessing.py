"""Preprocessing files before analyze

This script preprocess JSON files nad merge them into one JSON file. Each object in 
the JSON file contains a `data_origin` key containing the original filename of where 
the object came from

This script requires that `dvc.api`and `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * get_spreadsheet_cols - merge JSON files into one DataFrame
    * main - the main function of the script
"""

import dvc.api
import pandas as pd


def merge_json_files(start_file, file_names):
    """
    Merge JSON files into one DataFrame. Create new column containing 
    the original filename of where the row came from

    Args:
        start_file (str): Ścieżka do katalogu, w którym znajdują się pliki JSON.
        file_names (list): Lista nazw plików JSON, które mają zostać połączone.

    Returns:
        pd.DataFrame: Połączone dane w postaci obiektu DataFrame.
    """
    merged_data = pd.DataFrame()

    for file_name in file_names:
        data = pd.read_json(start_file + "/" + file_name, lines=True)
        data["data_origin"] = file_name
        merged_data = pd.concat([merged_data, data], ignore_index=True)

    merged_data = merged_data.reset_index(drop=True)
    return merged_data


def main():
    """
    Główna funkcja programu. Łączy pliki JSON i zapisuje wynik do pliku JSON.

    Pobiera parametry konfiguracyjne z DVC i wykonuje operacje na danych.
    """
    params = dvc.api.params_show()

    start_file = params["preprocessing"]["start_file"]
    target_file = params["preprocessing"]["target_file"]
    file_names = params["preprocessing"]["file_names"]

    merged_data = merge_json_files(start_file, file_names)
    merged_data.to_json(target_file, orient="records", lines=True)


if __name__ == "__main__":
    main()
