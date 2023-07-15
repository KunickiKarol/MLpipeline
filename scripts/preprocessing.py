from typing import List

import dvc.api
import pandas as pd
from pandas import DataFrame


def merge_json_files(start_file: str, file_names: List[str]) -> DataFrame:
    """
    Merge JSON files into one DataFrame. Create a new column containing
    the original filename of where the row came from

    Args:
        start_file (str): Path to the directory where the JSON files are located.
        file_names (list): List of JSON file names to be merged.

    Returns:
        pd.DataFrame: Merged data as a DataFrame object.
    """
    merged_data = pd.DataFrame()

    for file_name in file_names:
        data = pd.read_json(start_file + "/" + file_name, lines=True)
        data["data_origin"] = file_name
        merged_data = pd.concat([merged_data, data], ignore_index=True)

    merged_data = merged_data.reset_index(drop=True)
    return merged_data


def main() -> None:
    """
    Main function of the script. Merges JSON files and saves the result to a JSON file.

    Retrieves configuration parameters from DVC and performs data operations.
    """
    params = dvc.api.params_show()

    start_file = params["preprocessing"]["start_file"]
    target_file = params["preprocessing"]["target_file"]
    file_names = params["preprocessing"]["file_names"]

    merged_data = merge_json_files(start_file, file_names)
    merged_data.to_json(target_file, orient="records", lines=True)


if __name__ == "__main__":
    main()
