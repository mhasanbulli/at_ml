import os
import csv
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


class dataset:
    def __init__(self, data_dir="data/"):
        self.data_dir = Path(data_dir)

    def _get_data(self, directory):
        # This is not the best way to read the data as it iterates over the available data files.
        for file in (directory).iterdir():
            data = pd.read_csv(file, index_col="consumer_id")

        cols_with_na = [col for col in data.columns if data[col].isnull().any()]

        data.drop(cols_with_na + ["account_status"], axis=1, inplace=True)
        data_lof = pd.DataFrame(MinMaxScaler().fit_transform(data), index=data.index)
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.1).fit_predict(data_lof)
        data_lof.columns = data.columns
        data_lof["label"] = lof
        data_lof["label"] = data_lof["label"].replace([-1, 1], [1, 0])

        X = data_lof.drop(["label"], axis=1)
        y = data_lof["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        return X_train, X_test, y_train, y_test

    def get_data(self):
        return self._get_data(directory=self.data_dir)