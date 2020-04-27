import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pickle
import pandas as pd
import lightgbm as lgbm

from sklearn.metrics import classification_report

class lof_lgbm:
    def __init__(self):
        self.lgbm_model = None

    def train(self, X_train, y_train, X_test, y_test):
        self.lgbm_model = lgbm.LGBMClassifier(learning_rate=0.2, n_estimators=34, num_leaves=35)
        self.lgbm_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=["auc", "binary_logloss"]
        )

    def predict(self, X):
        y_pred = self.lgbm_model.predict(X, num_iteration=self.lgbm_model.best_iteration_)
        return y_pred

    def predict_prob(self, X):
        y_pred_p = self.lgbm_model.predict_proba(X)[:, 1]
        return y_pred_p

    def serialise(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self.lgbm_model, f)

    @staticmethod
    def deserialise(fname):
        model = lof_lgbm()
        with open(fname, "rb") as f:
            model.lgbm_model = pickle.load(f)
        return model