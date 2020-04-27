"""at_ml

Usage:
    atml-cli train <dataset_dir> <model_file>
    atml-cli predict <dataset_dir> <model_file>
    atml-cli (-h | --help)

Arguments:
    <dataset_dir>   Directory with either the sample data for training, or for prediction.
    <model_file>    Serialised model file.

Options:
    -h --help   Show this screen.
"""
import os

from docopt import docopt
from sklearn.metrics import classification_report

from at_ml import lof_lgbm, dataset



def train_model(dataset_dir, model_file):
    print(f"Training model from directory {dataset_dir}...")

    d_set = dataset(dataset_dir)
    X_train, X_test, y_train, y_test = d_set.get_data();

    model = lof_lgbm()
    model.train(X_train, y_train, X_test, y_test)

    print(f"Storing model to {model_file}.")
    model.serialise(model_file)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def predict(dataset_dir, model_file):
    print(f"Deserialising model {model_file}...")

    model = lof_lgbm.deserialise(model_file)

    y_pred = model.predict_results(dataset_dir)
    return y_pred

def main():
    arguments = docopt(__doc__)

    if arguments["train"]:
        train_model(
            arguments["<dataset_dir>"],
            arguments["<model_file>"]
        )
    elif arguments["predict"]:
        predict(
            arguments["<dataset_dir>"],
            arguments["<model_file>"]
        )

if __name__ == '__main__':
    main()