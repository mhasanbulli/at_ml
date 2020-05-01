import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import lightgbm as lgbm


get_ipython().run_line_magic("matplotlib", " inline")
pd.plotting.register_matplotlib_converters()
sns.set(style = "ticks")

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report  
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix


file_path = "../data/data.csv"
data = pd.read_csv(file_path, index_col = "consumer_id")

cols_with_na = [col for col in data.columns if data[col].isnull().any()]

data.drop(cols_with_na + ["account_status"], axis = 1, inplace = True)
data_lof = pd.DataFrame(MinMaxScaler().fit_transform(data), index = data.index)
lof = LocalOutlierFactor(n_neighbors=50, contamination=0.1).fit_predict(data_lof)
data_lof.columns = data.columns
data_lof["label"] = lof
data_lof["label"] = data_lof["label"].replace([-1,1],[1,0])
data_lof.head()


X = data_lof.drop(["label"], axis = 1)
y = data_lof["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
X_train.shape


est_model = lgbm.LGBMClassifier(learning_rate=0.1, n_estimators=20, num_leaves=5)

param_grid = {
    "n_estimators": [i for i in range(20, 40, 2)],
    "learning_rate": [0.10, 0.125, 0.15, 0.175, 0.2],
    "num_leaves": [i for i in range(5, 40, 5)]
}

grid_search = GridSearchCV(est_model, param_grid,cv = 10)

grid_search.fit(
    X_train, y_train,
    eval_set = [(X_test, y_test)],
    eval_metric = ["auc", "binary_logloss"],
    verbose = 0
)


print('Best parameters found by grid search are:', grid_search.best_params_)


lgmb_model = lgbm.LGBMClassifier(learning_rate=0.2, n_estimators=34, num_leaves=35)

lgmb_model.fit(
    X_train, y_train,
    eval_set = [(X_test, y_test)],
    eval_metric = ["auc", "binary_logloss"]
)


y_pred = lgmb_model.predict(X_test, num_iteration=lgmb_model.best_iteration_)

acc = accuracy_score(y_test, y_pred) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_test, y_pred) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_test, y_pred) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_test, y_pred) 
print("The F1-Score is {}".format(f1)) 
  
MCC = matthews_corrcoef(y_test, y_pred) 
print("The Matthew's correlation coefficient is {}".format(MCC)) 


y_pred_p = lgmb_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_p)

plt.figure(figsize = (8, 8))
plt.plot(fpr, tpr, color = "darkorange", lw = 2, label = 'ROC curve (area = get_ipython().run_line_magic("0.2f)'", " % roc_auc_score(y_test, y_pred_p))")
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.title('ROC curve for credit card defaulting classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("p_roc.png")


axis_labels = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (8, 8)) 
sns.heatmap(
    conf_matrix,
    xticklabels = axis_labels,  
    yticklabels = axis_labels,
    annot = True,
    fmt ="d"
) 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.savefig("p_cf.png")
