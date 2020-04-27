# at_ml
`at_ml` is a machine learning library that identifies fraudulent transactions 
using a gradient boosting framework that uses tree based learning algorithm, namely LightGBM.

## Installation

```bash
pip install at_ml
```

## Usage

There are two available modules in `at_ml`. 

`dataset` module can be used to load and transform the data.
This is also the module where unlabeled dataset is labeled using LOF method. `dataset` module produces four
dataset ready to be used in the model training.

`lof_lgbm` module is where the a LightGBM model is trained and can be used for predictions. We do not perform any hyperparameter optimisation
in this step. Parameters we are using in the model training are results of a grid search done prior to the 
model building.

An example of usage can be found below.
```python
from at_ml import dataset, lof_lgbm

d=dataset()
m=lof_lgbm()

X_train, X_test, y_train, y_test = d.get_data()
m.train(X_train, y_train, X_test, y_test)
```

## Usage - Command Line

`at_ml` comes with a command line CLI interface. You can get more information on how to use the CLI by typing
```bash
atml-cli --help
```
in the command line.

## Training Data
You can get the training data from [this link](https://at-ml.s3-ap-southeast-2.amazonaws.com/data.csv).
You can also use `download_data.sh` script to get a copy of the sample data. 

## Remarks & Assumptions

- We do not provide any hyperparamter optimisation in this package.
- We assume the training is being done on a dataset that has the similar characteristics to the 
provided sample data.
- We do not perform any missing data imputation as the provided sample data has no missing data.
- We do not provide any plotting functionality with this package.

## License

[MIT](https://choosealicense.com/licenses/mit/)

 

