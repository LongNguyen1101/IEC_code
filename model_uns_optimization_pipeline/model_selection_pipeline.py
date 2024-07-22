import os
import json
from model_uns_optimization_pipeline.optimize_params import KMeansPSO, BirchPSO, GaussianMixturesPSO, MiniBatchKMeansPSO, SpectralClusteringPSO
import pandas as pd

class RunPSO():
    def __init__(self, data_dir, dir_result='result_unsup', swarmsize:int=10, maxiter:int=10, verbose=False) -> None:
        self.data_dir = data_dir
        self.verbose = verbose
        self.dir_result = dir_result
        self.swarmsize = swarmsize
        self.maxiter = maxiter

    def get_data_name(self):
        try:
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv') and os.path.isfile(os.path.join(self.data_dir, f))]
            return csv_files
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
    
    def get_df_list(self):
        csv_files = self.get_data_name()
        dfs = {}
        for file in csv_files:
            dfs[file] = pd.read_csv(f'{self.data_dir}/{file}')
        
        return dfs
    
    def create_directory(self):
        try:
            os.mkdir(self.dir_result)
            print(f"Directory '{self.dir_result}' created successfully.")
        except OSError as e:
            print(f"Failed to create directory '{self.dir_result}': {e}")
    
    def process_data_with_pso(self, file_name, data):
        result = []

        print('KMeans')
        km_best_params, km_best_score = KMeansPSO(data, file_name, self.swarmsize, self.maxiter, verbose=self.verbose).run_pso()
        result.append({
            'model': 'KMeans',
            'params': km_best_params,
            'score': km_best_score
        })

        print('Gaussian mixtures')
        gm_best_params, gm_best_score = GaussianMixturesPSO(data, file_name, self.swarmsize, self.maxiter, verbose=self.verbose).run_pso()
        result.append({
            'model': 'Gaussian mixtures',
            'params': gm_best_params,
            'score': gm_best_score
        })

        print('Birch')
        bi_best_params, bi_best_score = BirchPSO(data, file_name, self.swarmsize, self.maxiter, verbose=self.verbose).run_pso()
        result.append({
            'model': 'Birch',
            'params': bi_best_params,
            'score': bi_best_score
        })

        print('Spectral clustering')
        sc_best_params, sc_best_score = SpectralClusteringPSO(data, file_name, self.swarmsize, self.maxiter, verbose=self.verbose).run_pso()
        result.append({
            'model': 'Spectral clustering',
            'params': sc_best_params,
            'score': sc_best_score
        })

        print('MiniBatch Kmeans')
        mk_best_params, mk_best_score = MiniBatchKMeansPSO(data, file_name, self.swarmsize, self.maxiter, verbose=self.verbose).run_pso()
        result.append({
            'model': 'MiniBatch Kmeans',
            'params': mk_best_params,
            'score': mk_best_score
        })
        
        file_name = file_name.replace('.csv', '.json')
        file_path = f'{self.dir_result}/{file_name}'

        try:
            with open(file_path, 'w') as file:
                json.dump(result, file, indent=4)
        except IOError as e:
            print(f"Failed to write data to {file_path}: {e}")
        

    def run(self):
        dfs = self.get_df_list()
        self.create_directory()

        for file_name, data in dfs.items():
            print(f'Optimize data: {file_name}')
            self.process_data_with_pso(file_name, data)
            print(f'Optimize successfull: {file_name}')
            
            if self.verbose:
                print(f'{file_name} optimized successfully')