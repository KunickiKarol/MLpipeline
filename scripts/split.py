import json
import os
from typing import Tuple

import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split


def split_json_file(
    start_file: str,
    target_file_train: str,
    target_file_test: str,
    class_column_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_json(start_file, lines=True)

    train, test = train_test_split(df, test_size=0.2, stratify=df[class_column_name])

    return train, test


def main() -> None:
    params = dvc.api.params_show()

    start_file = params["preprocessing"]["target_file"]
    target_file_train = params["split"]["target_file_train"]
    target_file_test = params["split"]["target_file_test"]
    class_column_name = params["class_column_name"]

    train, test = split_json_file(
        start_file, target_file_train, target_file_test, class_column_name
    )
    train.to_json(target_file_train, orient="records", lines=True)
    test.to_json(target_file_test, orient="records", lines=True)


if __name__ == "__main__":
    main()
