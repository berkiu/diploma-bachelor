from __future__ import annotations
import numpy as np
import pandas as pd


def my_train_test_split(data: np.ndarray, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    split_point = int(len(data) * test_size)
    df_train_val = np.array(data[0:split_point], dtype=object)
    df_test = np.array(data[split_point:], dtype=object)
    return df_train_val, df_test


def data_preprocessing(data: pd.core.frame.DataFrame, type: str) -> pd.core.frame.DataFrame:
    data_preprocessed = data.copy()

    if type == 'law':
        # change labels to int law
        data_preprocessed.loc[data_preprocessed['label'] == 'header', 'label'] = 0
        data_preprocessed.loc[data_preprocessed['label'] == 'structure_unit', 'label'] = 1
        data_preprocessed.loc[data_preprocessed['label'] == 'raw_text', 'label'] = 2
        data_preprocessed.loc[data_preprocessed['label'] == 'footer', 'label'] = 3
        data_preprocessed.loc[data_preprocessed['label'] == 'application', 'label'] = 4

    elif type == 'tz':
        # change labels to int tz
        data_preprocessed.loc[data_preprocessed['label'] == 'item', 'label'] = 0
        data_preprocessed.loc[data_preprocessed['label'] == 'part', 'label'] = 1
        data_preprocessed.loc[data_preprocessed['label'] == 'raw_text', 'label'] = 2
        data_preprocessed.loc[data_preprocessed['label'] == 'title', 'label'] = 3
        data_preprocessed.loc[data_preprocessed['label'] == 'toc', 'label'] = 4

    data_preprocessed = data_preprocessed.astype({'label': 'int32'})
    # drop uid and text columns
    data_preprocessed = data_preprocessed.drop(['uid', 'text'], axis=1)
    print(f'{type} data successfully preprocessed')
    return data_preprocessed


def find_split_points(data: pd.core.frame.DataFrame) -> np.ndarray:
    # find all points where new doc in data starts for next splitting
    split_points = []
    prev = 0
    for i in range(len(data.group)):
        if data.group[i] != prev:
            split_points.append(i)
        prev = data.group[i]
    split_points.append(len(data.group))
    return np.array(split_points)


def split_data_by_docs(data: pd.core.frame.DataFrame) -> np.ndarray:
    # get split points
    split_points = find_split_points(data)
    # split data by every doc
    dfs_of_every_doc = []
    for i in range(1, len(split_points)):
        start, end = split_points[i - 1], split_points[i]
        df = data[start:end].copy()
        df = df.drop(['group'], axis=1)
        dfs_of_every_doc.append(df)
    return np.array(dfs_of_every_doc)
