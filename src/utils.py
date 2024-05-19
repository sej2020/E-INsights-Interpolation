import pathlib
import pandas as pd
import numpy as np

def searching_all_files(directory: str) -> list:
    """
    Searches all files in a directory and returns a list of all files in the directory

    Args:
        directory: path to directory

    Returns:
        list of all files in the directory
    """
    dirpath = pathlib.Path(directory)
    assert dirpath.is_dir()
    file_list = []
    for x in dirpath.iterdir():
        if x.is_file():
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(searching_all_files(x))
    return file_list


def filter_data(path: pathlib.Path | str, max_missing_interval: int = 360, min_density: float = 0.75):
    """
    Filters data by emitting a boolean value based on whether a.) the data has a missing interval longer
    than max_missing_interval, and b.) the data has a density greater than min_density

    Args:
        path: path to csv file
        max_missing_interval: maximum length of missing interval allowed
        min_density: minimum density of data allowed

    Returns:
        boolean value based on whether the data has a missing interval longer than max_missing_interval,
        and whether the data has a density greater than min_density
    """
    df = pd.read_csv(path)
    missing_int_length = 0
    missing_val_total = 0
    for ind in df.index:
        if np.isnan(df.iloc[ind, -1]):
            missing_int_length += 1
            missing_val_total += 1
        else:
            missing_int_length = 0
        if missing_int_length > max_missing_interval:
            return False
    if missing_val_total / len(df) > min_density:
        return False
    else:
        return True