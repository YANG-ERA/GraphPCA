
import numpy as np
import scipy
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.sparse import issparse
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances as pair
from sklearn.metrics import adjusted_rand_score as ari_score
from .utils import *

def set_n_neighbors(platform, n_neighbors):
    if n_neighbors is not None:
        return n_neighbors

    if platform == "Visium":
        return 6
    elif platform == "ST":
        return 4
    else:
        raise ValueError("Please specify n_neighbors value or use valid platform value, like 'Visium' or 'ST'. ")


def Run_GPCA(adata, location=None, network=None, n_components=50, method="knn", platform="Visium", _lambda=0.5,
             n_neighbors=6, save_reconstruction=False):

    global graphL
    Expr = adata.X
    if issparse(Expr):
        Expr = Expr.todense()
    if network is not None:
        graph = network
    else:
        n_neighbors = set_n_neighbors(platform, n_neighbors)
        if method == "knn":
            graph = kneighbors_graph(np.asarray(location), int(n_neighbors), metric='euclidean',
                                     metric_params={}, include_self=False)
            graph = 0.5 * (graph + graph.T)
    graphL = csgraph.laplacian(graph, normed=False)

    G = scipy.sparse.eye(adata.shape[0]) + _lambda * graphL
    if issparse(G):
        Ginv = np.array(np.linalg.inv(G.todense()))
    else:
        Ginv = np.array(np.linalg.inv(G))
    C = np.dot(np.dot(Expr.T, Ginv), Expr)
    lambdas, W = np.linalg.eigh(C)
    W = W[:, ::-1]
    W = W[:, :n_components]
    Z = np.dot(np.dot(Ginv, Expr), W)
    if save_reconstruction:
        adata.uns["GraphPCA_ReX"] = np.dot(Z,W.T)
    return Z, W


def consensus_gpca(adata, location, n_components=50, method="knn", platform="Visium",
                   lambda_list=np.arange(0, 0.5, 0.1), n_neighbors=6,
                   n_clusters=8, kmeans_seed=430):
    n = adata.X.shape[0]
    sk = AgglomerativeClustering(n_clusters, linkage='average')
    matrix_sum = np.zeros(shape=(n, n))
    for _lambda in lambda_list:
        print("lambda: ", _lambda)
        Z = Run_GPCA(adata, location=location, n_components=n_components, method=method,
                     n_neighbors=n_neighbors, platform=platform, _lambda=_lambda)
        estimator = KMeans(n_clusters=n_clusters, random_state=kmeans_seed)
        res = estimator.fit(Z[:, :])
        lable_pred = res.labels_
        temp = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n)[i + 1:]:
                if lable_pred[i] == lable_pred[j]:
                    temp[i, j] = temp[j, i] = 1
        matrix_sum += temp
    matrix_mean = matrix_sum / (len(lambda_list))
    sk.fit(matrix_mean)
    labels = sk.labels_
    return labels
