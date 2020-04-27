"""at_ml

Usage:
    atml-cli train <dataset-dir> <model-file>
    atml-cli predict <model-file>
    atml-cli (-h | --help)

Arguments:
    <dataset-dir>   Directory with the dataset.
    <model-file>    Serialised model file.

Options:
    -h --help   Show this screen.
"""
import os

from docopt import docopt
from sklearn.metrics import classification_report

from at_ml import lof_lgbm, dataset



def train_model(dataset_dir, model_file):
    print(f"Training model from directory {dataset_dir}")

    d_set = dataset(dataset_dir)
    X_train, X_test, y_train, y_test = d_set.get_data();

    model = lof_lgbm()
    model.train(X_train, y_train, X_test, y_test)

    print(f"Storing model to {model_file}")
    model.serialise(model_file)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def main():
    arguments = docopt(__doc__)

    if arguments['train']:
        train_model(arguments['<dataset-dir>'],
                    arguments['<model-file>']
        )

if __name__ == '__main__':
    main()