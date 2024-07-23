import numpy as np

from sklearn.cluster import KMeans, Birch, SpectralClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import adjusted_rand_score
from pyswarm import pso

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class KMeansPSO():
    def __init__(self, data, file_name, swarmsize: int=10, maxiter: int=10, n_cluster: int=3, verbose=False) -> None:
        self.lb = [0, 1, 100, 0.0001, 0, 1] 
        self.ub = [1, 10, 1000, 1.0, 1, 100]
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.file_name = file_name
        self.n_cluster = n_cluster
    
    def map_init(self, value):
        return ['k-means++', 'random'][value]
    
    def map_algorithm(self, value):
        return ['lloyd', 'elkan'][value]

    def evaluate(self, params):
        ari_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            cluster = KMeans(n_clusters=self.n_cluster,
                             init=self.map_init(int(params[0])),
                             n_init=int(params[1]),
                             max_iter=int(params[2]),
                             tol=params[3],
                             algorithm=self.map_algorithm(int(params[4])),
                             random_state=int(params[5]))

            y_pred = cluster.fit_predict(X_train)
            ari_train = adjusted_rand_score(y_train, y_pred)

            y_pred_test = cluster.fit_predict(X_test)
            ari_test = adjusted_rand_score(y_test, y_pred_test)

            if ari_train > ari_test:
                ari_scores.append(ari_train)
            else: ari_scores.append(ari_test)
        
        mean_ari = sum(ari_scores) / 5
        
        if self.verbose:  print(f'{self.file_name}: {mean_ari}')
        
        return -mean_ari  
    
    def run_pso(self):
        best_params, best_score = pso(self.evaluate, self.lb, self.ub, swarmsize=self.swarmsize, maxiter=self.maxiter)

        best_params = {
            'n_clusters': self.n_cluster,
            'init': self.map_init(int(best_params[0])),
            'n_init': int(best_params[1]),
            'max_iter': int(best_params[2]),
            'tol': best_params[3],
            'algorithm': self.map_algorithm(int(best_params[4])),
            'random_state': int(best_params[5])
        }

        return best_params, -best_score
    
class GaussianMixturesPSO():
    def __init__(self, data, file_name, swarmsize: int=10, maxiter: int=10, n_cluster: int=3, verbose=False) -> None:
        self.lb = [0, 0.001, 0.00001, 100, 1, 0, 1]
        self.ub = [3, 1.0, 1.0, 500, 20, 3, 100]
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.file_name = file_name
        self.n_cluster = n_cluster
    
    def map_covariance_type(self, value):
        return ['full', 'tied', 'diag', 'spherical'][value]
    
    def map_init_params(self, value):
        return ['kmeans', 'k-means++', 'random', 'random_from_data'][value]

    def evaluate(self, params):
        ari_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            cluster = GaussianMixture(n_components=self.n_cluster,
                                      covariance_type=self.map_covariance_type(int(params[0])),
                                      tol=params[1],
                                      reg_covar=params[2],
                                      max_iter=int(params[3]),
                                      n_init=int(params[4]),
                                      init_params=self.map_init_params(int(params[5])),
                                      random_state=int(params[6]))


            y_pred = cluster.fit_predict(X_train)
            ari_train = adjusted_rand_score(y_train, y_pred)

            y_pred_test = cluster.fit_predict(X_test)
            ari_test = adjusted_rand_score(y_test, y_pred_test)

            if ari_train > ari_test:
                ari_scores.append(ari_train)
            else: ari_scores.append(ari_test)
        
        mean_ari = sum(ari_scores) / 5
        
        if self.verbose:  print(f'{self.file_name}: {mean_ari}')
        
        return -mean_ari
    
    def run_pso(self):
        best_params, best_score = pso(self.evaluate, self.lb, self.ub, swarmsize=self.swarmsize, maxiter=self.maxiter)

        best_params = {
            'n_components': self.n_cluster,
            'covariance_type': self.map_covariance_type(int(best_params[0])),
            'tol': best_params[1],
            'reg_covar': best_params[2],
            'max_iter': int(best_params[3]),
            'n_init': int(best_params[4]),
            'init_params': self.map_init_params(int(best_params[5])),
            'random_state': int(best_params[6])
        }

        return best_params, -best_score
    
class BirchPSO():
    def __init__(self, data, file_name, swarmsize: int=10, maxiter: int=10, n_cluster: int=3, verbose=False) -> None:
        self.lb = [0.05, 2]
        self.ub = [0.95, 100]
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.file_name = file_name
        self.n_cluster = n_cluster

    def evaluate(self, params):
        ari_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            cluster = Birch(n_clusters=self.n_cluster,
                            threshold=params[0],
                            branching_factor=int(params[1]))

            y_pred = cluster.fit_predict(X_train)
            ari_train = adjusted_rand_score(y_train, y_pred)

            y_pred_test = cluster.fit_predict(X_test)
            ari_test = adjusted_rand_score(y_test, y_pred_test)

            if ari_train > ari_test:
                ari_scores.append(ari_train)
            else: ari_scores.append(ari_test)
        
        mean_ari = sum(ari_scores) / 5
        
        if self.verbose:  print(f'{self.file_name}: {mean_ari}')
        
        return -mean_ari
    
    def run_pso(self):
        best_params, best_score = pso(self.evaluate, self.lb, self.ub, swarmsize=self.swarmsize, maxiter=self.maxiter)

        best_params = {
            'n_clusters': self.n_cluster,
            'threshold': best_params[0],
            'branching_factor': int(best_params[1]),
        }

        return best_params, -best_score

class SpectralClusteringPSO():
    def __init__(self, data, file_name, swarmsize: int=10, maxiter: int=10, n_cluster: int=3, verbose=False) -> None:
        self.lb = [0, 1, 1, 1.0, 1, 0, 0, 1.0, 1]
        self.ub = [1, 5, 10, 2.0, 5, 2, 5, 2.0, 100]
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.file_name = file_name
        self.n_cluster = n_cluster
    
    def map_eigen_solver(self, value):
        return ['arpack', 'amg'][value]
    
    def map_assign_labels(self, value):
        return ['kmeans', 'discretize', 'cluster_qr'][value]

    def evaluate(self, params):
        ari_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            cluster = SpectralClustering(
                n_clusters=self.n_cluster,
                eigen_solver=self.map_eigen_solver(int(params[0])),
                n_components=int(params[1]),
                n_init=int(params[2]),
                gamma=params[3],
                n_neighbors=int(params[4]),
                assign_labels=self.map_assign_labels(int(params[5])),
                degree=int(params[6]),
                coef0=params[7],
                random_state=int(params[8])
            )

            y_pred = cluster.fit_predict(X_train)
            ari_train = adjusted_rand_score(y_train, y_pred)

            y_pred_test = cluster.fit_predict(X_test)
            ari_test = adjusted_rand_score(y_test, y_pred_test)

            if ari_train > ari_test:
                ari_scores.append(ari_train)
            else: ari_scores.append(ari_test)
        
        mean_ari = sum(ari_scores) / 5
        
        if self.verbose:  print(f'{self.file_name}: {mean_ari}')
        
        return -mean_ari
        
    
    def run_pso(self):
        best_params, best_score = pso(self.evaluate, self.lb, self.ub, swarmsize=self.swarmsize, maxiter=self.maxiter)

        best_params = {
            'n_clusters': self.n_cluster,
            'eigen_solver': self.map_eigen_solver(int(best_params[0])),
            'n_components': int(best_params[1]),
            'n_init': int(best_params[2]),
            'gamma': best_params[3],
            'n_neighbors': int(best_params[4]),
            'assign_labels': self.map_assign_labels(int(best_params[5])),
            'degree': best_params[6],
            'coef0': best_params[7],
            'random_state': int(best_params[8])
        }

        return best_params, -best_score
    
class MiniBatchKMeansPSO():
    def __init__(self, data, file_name, swarmsize: int=10, maxiter: int=10, n_cluster: int=3, verbose=False) -> None:
        self.lb = [0, 100, 1024, 0.0, 5, 4, 1, 0.01, 1]
        self.ub = [1, 500, 4096, 1.0, 20, 20, 10, 1.0, 100]
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.file_name = file_name
        self.n_cluster = n_cluster

    def map_init(self, value):
        return ['k-means++', 'random'][value]
    
    def evaluate(self, params):
        ari_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            cluster = MiniBatchKMeans(
                n_clusters=self.n_cluster,
                init=self.map_init(int(params[0])),
                max_iter=int(params[1]),
                batch_size=int(params[2]),
                tol=params[3],
                max_no_improvement=int(params[4]),
                init_size=int(params[5]),
                n_init=int(params[6]),
                reassignment_ratio=params[7],
                random_state=int(params[8])
            )

            y_pred = cluster.fit_predict(X_train)
            ari_train = adjusted_rand_score(y_train, y_pred)

            y_pred_test = cluster.fit_predict(X_test)
            ari_test = adjusted_rand_score(y_test, y_pred_test)

            if ari_train > ari_test:
                ari_scores.append(ari_train)
            else: ari_scores.append(ari_test)
        
        mean_ari = sum(ari_scores) / 5
        
        if self.verbose:  print(f'{self.file_name}: {mean_ari}')
        
        return -mean_ari
        
    
    def run_pso(self):
        best_params, best_score = pso(self.evaluate, self.lb, self.ub, swarmsize=self.swarmsize, maxiter=self.maxiter)

        best_params = {
            'n_clusters': self.n_cluster,
            'init': self.map_init(int(best_params[0])),
            'max_iter': int(best_params[1]),
            'batch_size': int(best_params[2]),
            'tol': best_params[3],
            'max_no_improvement': int(best_params[4]),
            'init_size': int(best_params[5]),
            'n_init': int(best_params[6]),
            'reassignment_ratio': best_params[7],
            'random_state': int(best_params[8])
        }

        return best_params, -best_score