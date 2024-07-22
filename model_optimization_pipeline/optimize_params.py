import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from pyswarm import pso

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def sigmoid(x, k=80):
    return 1 / (1 + np.exp(-k*x))

def smooth_sigmoid_y(x):
    sigmoid_value = sigmoid(x - 0.95)
    return -(1 - x) * sigmoid_value - x * (1 - sigmoid_value)

class RandomForestPSO():
    def __init__(self, data, file_name, swarmsize: int=10, maxiter: int=10, verbose=False) -> None:
        self.lb = [10, 1, 2, 1, 0.0, 0.0, 0.0, 0, 0] 
        self.ub = [200, 50, 20, 20, 0.5, 1.0, 1.0, 1, 2]
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.file_name = file_name

    def map_criterion(slef, value):
        return ["gini", "entropy"][value]

    def map_max_features(self, value):
        return [None, "sqrt", "log2"][value]

    def evaluate(self, params):
        n_estimators = int(params[0])
        max_depth = int(params[1])
        min_samples_split = int(params[2])
        min_samples_leaf = int(params[3])
        min_weight_fraction_leaf = params[4]
        min_impurity_decrease = params[5]
        ccp_alpha = params[6]
        criterion = self.map_criterion(int(params[7]))
        max_features = self.map_max_features(int(params[8]))

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            criterion=criterion,
            max_features=max_features,
            random_state=42
        )

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)])

        scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='accuracy')
        smooth = smooth_sigmoid_y(scores.mean()) # smooth <= 0

        if self.verbose: print(f'{self.file_name}: {-smooth}')

        return smooth
    
    def run_pso(self):
        best_params, best_score = pso(self.evaluate, self.lb, self.ub, swarmsize=self.swarmsize, maxiter=self.maxiter)

        best_params = {
            'n_estimators': int(best_params[0]),
            'max_depth': int(best_params[1]),
            'min_samples_split': int(best_params[2]),
            'min_samples_leaf': int(best_params[3]),
            'min_weight_fraction_leaf': best_params[4],
            'min_impurity_decrease': best_params[5],
            'ccp_alpha': best_params[6],
            'criterion': self.map_criterion(int(best_params[7])),
            'max_features': self.map_max_features(int(best_params[8]))
        }

        return best_params, -best_score # return smooth to >= 0
    

class SVCPSO():
    def __init__(self, data, file_name, swarmsize: int=10, maxiter: int=10, verbose=False) -> None:
        self.lb = [0.1, 0.01, 1, 0, 0.0, 0]
        self.ub = [10, 1, 5, 3, 10, 1]  
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.file_name = file_name

    def map_kernel(self, value):
        return ["linear", "poly", "rbf", "sigmoid"][value]
    
    def map_shrinking(self, value):
        return [True, False][value]

    def evaluate(self, params):
        C = params[0]
        gamma = params[1]
        degree = int(params[2])
        kernel = self.map_kernel(int(params[3]))
        coef0 = params[4]
        shrinking = self.map_shrinking(int(params[5]))
        
        clf = SVC(
            C=C,
            gamma=gamma,
            degree=degree,
            kernel=kernel,
            coef0=coef0,
            shrinking=shrinking,
            random_state=42
        )

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)])

        scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='accuracy')
        smooth = smooth_sigmoid_y(scores.mean())

        if self.verbose: print(f'{self.file_name}: {-smooth}')

        return smooth
    
    def run_pso(self):
        best_params, best_score = pso(self.evaluate, self.lb, self.ub, swarmsize=self.swarmsize, maxiter=self.maxiter)

        best_params = {
            'C': best_params[0],
            'gamma': best_params[1],
            'degree': int(best_params[2]),
            'kernel': self.map_kernel(int(best_params[3])),
            'coef0': best_params[4],
            'shrinking': self.map_shrinking(int(best_params[5]))
        }

        return best_params, -best_score

class KNNPSO():
    def __init__(self, data, file_name, swarmsize: int=10, maxiter: int=10, verbose=False) -> None:
        self.lb = [1, 1, 0, 0, 1]
        self.ub = [10, 5, 1, 1, 100]
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.file_name = file_name

    def map_weights(self, value):
        return ["uniform", "distance"][value]
    
    def map_algorithm(self, value):
        return ["auto", "ball_tree", "kd_tree", "brute"][value]

    def evaluate(self, params):
        n_neighbors = int(params[0])
        p = int(params[1])
        weights = self.map_weights(int(params[2]))
        algorithm = self.map_algorithm(int(params[3]))
        leaf_szie = int(params[4])

        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            p=p,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_szie
        )

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)])

        scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='accuracy')
        smooth = smooth_sigmoid_y(scores.mean())

        if self.verbose: print(f'{self.file_name}: {-smooth}')

        return smooth
    
    def run_pso(self):
        best_params, best_score = pso(self.evaluate, self.lb, self.ub, swarmsize=self.swarmsize, maxiter=self.maxiter)
        
        best_params = {
            'n_neighbors': int(best_params[0]),
            'p': int(best_params[1]),
            'weights': self.map_weights(int(best_params[2])),
            'algorithm': self.map_algorithm(int(best_params[3])),
            'leaf_size': int(best_params[4])
        }
        
        return best_params, -best_score
    
class XGBoostPSO():
    def __init__(self, data, file_name, swarmsize: int=10, maxiter: int=10, verbose=False) -> None:
        self.lb = [0.01, 1, 1, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ub = [1.0, 1000, 100, 1, 1, 1.0, 1.0, 1.0, 1.0, 100]  
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.file_name = file_name

    def map_booster(self, value):
        return ["gbtree", "gblinear", "dart"][value]

    def evaluate(self, params):
        learning_rate = params[0]
        n_estimators = int(params[1])
        max_depth = int(params[2])
        booster = self.map_booster(int(params[3]))
        colsample_bytree = params[4]
        colsample_bylevel = params[5]
        colsample_bynode = params[6]
        subsample = params[7]
        eta = params[8]
        gamma = params[9]

        clf = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            booster=booster,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            subsample=subsample,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            eta=eta,
            gamma=gamma
        )

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)])

        scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='accuracy')
        smooth = smooth_sigmoid_y(scores.mean())

        if self.verbose: print(f'{self.file_name}: {-smooth}')

        return smooth
    
    def run_pso(self):
        best_params, best_score = pso(self.evaluate, self.lb, self.ub, swarmsize=self.swarmsize, maxiter=self.maxiter)

        best_params = {
            'learning_rate': best_params[0],
            'n_estimators': int(best_params[1]),
            'max_depth': int(best_params[2]),
            'booster': self.map_booster(int(best_params[3])),
            'colsample_bytree': best_params[4],
            'colsample_bylevel': best_params[5],
            'colsample_bynode': best_params[6],
            'subsample': best_params[7],
            'eta': best_params[8],
            'gamma': best_params[9]
        }

        return best_params, -best_score

class DecisionTreePSO():
    def __init__(self, data, file_name, swarmsize: int=10, maxiter: int=10, verbose=False) -> None:
        self.lb = [1, 0.05, 0.0, 0, 0, 0]
        self.ub = [100, 1.0, 0.5, 2, 1, 1]  
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.file_name = file_name

    def map_criterion(self, value):
        return ["gini", "entropy"][value]

    def map_splitter(self, value):
        return ["best", "random"][value]

    def map_max_features(self, value):
        return [None, "sqrt", "log2"][value]

    def evaluate(self, params):
        max_depth = int(params[0])
        min_samples_split = params[1]
        min_weight_fraction_leaf = params[2]
        max_features = self.map_max_features(int(params[3]))
        criterion = self.map_criterion(int(params[4]))
        splitter = self.map_splitter(int(params[5]))

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            criterion=criterion,
            splitter=splitter,
            random_state=42
        )

        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)])
        
        scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='accuracy')
        smooth = smooth_sigmoid_y(scores.mean())

        if self.verbose: print(f'{self.file_name}: {-smooth}')

        return smooth
    
    def run_pso(self):
        best_params, best_score = pso(self.evaluate, self.lb, self.ub, swarmsize=self.swarmsize, maxiter=self.maxiter)

        best_params = {
            'max_depth': int(best_params[0]),
            'min_samples_split': best_params[1],
            'min_weight_fraction_leaf': best_params[2],
            'max_features': self.map_max_features(int(best_params[3])),
            'criterion': self.map_criterion(int(best_params[4])),
            'splitter': self.map_splitter(int(best_params[5]))
        }

        return best_params, -best_score
