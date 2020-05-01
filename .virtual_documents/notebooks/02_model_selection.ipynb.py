import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time


get_ipython().run_line_magic("matplotlib", " inline")
pd.plotting.register_matplotlib_converters()

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from sklearn import metrics

sns.set(style = "ticks")


file_path = "../data/data.csv"
data = pd.read_csv(file_path, index_col = "consumer_id")

cols_with_na = [col for col in data.columns if data[col].isnull().any()]

data.drop(cols_with_na + ["account_status"], axis = 1, inplace = True)


def tsne_plot(d, m, n_color):
    """
    Function to visualise clusters with t-SNE for a given dataset d and model m
    """
    d_copy = d.copy()
    
    d_copy["tsne-d1"] = m[:, 0]
    d_copy["tsne-d2"] = m[:, 1]

    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x="tsne-d1", y="tsne-d2",
        palette=sns.color_palette("hls", n_color),
        hue = "label",
        data=d_copy,
        legend="full",
        alpha=0.3
    )


n_clusters = 2
data_norm_k = pd.DataFrame(MinMaxScaler().fit_transform(data), index = data.index)
k_means = KMeans(n_clusters=n_clusters, random_state=123, init="k-means++").fit_predict(data_norm_k)
data_norm_k.columns = data.columns
data_norm_k["label"] = k_means + 1
data_norm_k["label"] = data_norm_k["label"].apply(lambda i: str(i))


score = metrics.silhouette_score(data_norm_k[data.columns], data_norm_k["label"])
print("For n_clusters = {}, silhouette score is {}.".format(n_clusters, score))


start_time = time.time()
tsne = TSNE(n_components = 2, perplexity = 30, n_iter = 500, verbose = 0, random_state = 123, learning_rate=50)
tsne_results = tsne.fit_transform(data_norm_k[data.columns])

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-start_time))

tsne_plot(data_norm_k, tsne_results, 2)
#plt.savefig("p_kmeans.png")


data_norm_db = pd.DataFrame(MinMaxScaler().fit_transform(data), index = data.index)
db_scan = DBSCAN(eps = 0.99, min_samples = 100, metric = "euclidean", n_jobs = -1).fit(data_norm_db)
data_norm_db.columns = data.columns
data_norm_db["label"] = db_scan.labels_
data_norm_db["label"] = data_norm_db["label"].apply(lambda i: str(i))
data_norm_db.head()


score_db = metrics.silhouette_score(data_norm_db[data.columns], data_norm_db["label"])
n_clusters_ = len(set(db_scan.labels_)) - (1 if -1 in db_scan.labels_ else 0)
print("For n_clusters = {}, silhouette score is {}.".format(n_clusters_, score_db))


tsne_plot(data_norm_db, tsne_results, n_clusters_ + 1)


#data_lof = data.copy()
data_lof = pd.DataFrame(MinMaxScaler().fit_transform(data), index = data.index)
lof = LocalOutlierFactor(n_neighbors=50, contamination=0.1).fit_predict(data_lof)
data_lof.columns = data.columns
data_lof["label"] = lof
data_lof["label"] = data_lof["label"].apply(lambda i: str(i))
data_lof.head()


start_time = time.time()
tsne = TSNE(n_components = 2, perplexity = 50, n_iter = 500, verbose = 0, random_state = 123, learning_rate=50)
tsne_results_lof = tsne.fit_transform(data_lof[data.columns])

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-start_time))

tsne_plot(data_lof, tsne_results_lof, 2)
#plt.savefig("p_lot.png")


perc_lof = ((data_lof["label"] == "-1").sum() / len(data_lof["label"])) * 100
perc_kmeans = ((data_norm_k["label"] == "2").sum() / len(data_norm_k["label"])) * 100

print("LOF classifies {}% of the sample data as fraudulent whereas k-means classifies {:2.2f}% of cases as fraudulent.".format(perc_lof, perc_kmeans))
