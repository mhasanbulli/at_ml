import pickle
import pandas as pd
import lightgbm as lgbm

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

class lof_lgbm:
    def __init__(self):
        self.lgmb_model = None

    def train(self, X_train, y_train):
        self.lgbm_model = lgbm.LGBMClassifier(learning_rate=0.2, n_estimators=34, num_leaves=35)
        self.lgmb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=["auc", "binary_logloss"]
        )

    def predict(self, X):
        y_pred = self.lgmb_model.predict(X, num_iteration=self.lgmb_model.best_iteration_)
        return y_pred

    def predict_prob(self, X):
        y_pred_p = self.lgmb_model.predict_proba(X)[:, 1]
        return y_pred_p

    def serialise(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self.lgmb, f)

    @staticmethod
    def deserialise(fname):
        model = lof_lgbm()
        with open(fname, "rb") as f:
            model.lgmb = pickle.load(f)
        return model