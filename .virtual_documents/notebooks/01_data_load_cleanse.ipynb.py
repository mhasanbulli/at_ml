import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", " inline")
pd.plotting.register_matplotlib_converters()

file_path = "../data/data.csv"
data = pd.read_csv(file_path, index_col = "consumer_id")


data.head()


obj_cols = [col for col in data.columns if data[col].dtype == "object"]
num_cols = [col for col in data.columns if data[col].dtype in ["int64", "float64"]]


print(num_cols, obj_cols)


obj_cols_with_na = [col for col in obj_cols if data[col].isnull().any()]
num_cols_with_na = [col for col in num_cols if data[col].isnull().any()]

print(obj_cols_with_na, num_cols_with_na)


def na_ratio(col):
    ratio = col.isnull().sum() / len(col)
    if (ratio > 0):
        return ratio
    
data[obj_cols_with_na + num_cols_with_na].apply(na_ratio)


data.drop(obj_cols_with_na + num_cols_with_na + ["account_status"], axis = 1, inplace = True)

data.head()
