import os
import json
from model_optimization_pipeline.optimize_params import RandomForestPSO, KNNPSO, XGBoostPSO, SVCPSO, DecisionTreePSO
import pandas as pd

class RunPSO():
    def __init__(self, data_dir, dir_result='result_sup', swarmsize:int=10, maxiter:int=10, verbose=False) -> None:
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

        print('Decision Tree')
        dt_best_params, dt_best_score = DecisionTreePSO(data, file_name, self.swarmsize, self.maxiter, verbose=self.verbose).run_pso()
        result.append({
            'model': 'Decision Tree',
            'params': dt_best_params,
            'score': dt_best_score
        })

        print('Random Forest')
        rf_best_params, rf_best_score = RandomForestPSO(data, file_name, self.swarmsize, self.maxiter, verbose=self.verbose).run_pso()
        result.append({
            'model': 'Random Forest',
            'params': rf_best_params,
            'score': rf_best_score
        })

        print('KNN')
        knn_best_params, knn_best_score = KNNPSO(data, file_name, self.swarmsize, self.maxiter, verbose=self.verbose).run_pso()
        result.append({
            'model': 'KNN',
            'params': knn_best_params,
            'score': knn_best_score
        })

        print('XGBoost')
        xg_best_params, xg_best_score = XGBoostPSO(data, file_name, self.swarmsize, self.maxiter, verbose=self.verbose).run_pso()
        result.append({
            'model': 'XGBoost',
            'params': xg_best_params,
            'score': xg_best_score
        })

        print('SVC')
        svc_best_params, svc_best_score = SVCPSO(data, file_name, self.swarmsize, self.maxiter, verbose=self.verbose).run_pso()
        result.append({
            'model': 'SVC',
            'params': svc_best_params,
            'score': svc_best_score
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