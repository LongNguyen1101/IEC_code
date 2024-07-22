import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class ScalePipeline:
    
    def __init__(self, config) -> None:
        self.impute_path = config['directory_impute']
        self.scale_path = config['directory_scale']
        self.metadata_filename = config['metadata']
        self.dfs = []

    def write_metadata(self, directory, content):
        filetxt = os.path.join(directory, self.metadata_filename)
        if content == '':
          with open(filetxt, 'w') as file:
            file.write(content)
        else:
            with open(filetxt, 'a') as file:
                file.write(content + '\n')

    def get_df_list(self):
        dfs = []
        directory = self.impute_path
        for i in range(7):
            for j in range(2):
                dfs.append(pd.read_csv(f'{directory}/{i + 1}_{j + 1}.csv'))

        self.dfs = dfs

    def min_max_scaler(self, df, df_name):
        X = df.drop('label', axis=1)
        y = df['label']

        min_max_scaler = MinMaxScaler()
        df_min_max_scaled = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)
        df_min_max_scaled['label'] = y

        filepath = os.path.join(self.scale_path, f'{df_name}_1.csv')
        df_min_max_scaled.to_csv(filepath, index=False)

    def standard_scaler(self, df, df_name):
        X = df.drop('label', axis=1)
        y = df['label']

        standard_scaler = StandardScaler()
        df_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(X), columns=X.columns)
        df_standard_scaled['label'] = y

        filepath = os.path.join(self.scale_path, f'{df_name}_2.csv')
        df_standard_scaled.to_csv(filepath, index=False)

    def df_robust_scaled(self, df, df_name):
        X = df.drop('label', axis=1)
        y = df['label']

        robust_scaler = RobustScaler()
        df_robust_scaled = pd.DataFrame(robust_scaler.fit_transform(X), columns=X.columns)
        df_robust_scaled['label'] = y

        filepath = os.path.join(self.scale_path, f'{df_name}_3.csv')
        df_robust_scaled.to_csv(filepath, index=False)

    def scaler(self):
        if not os.path.exists(self.scale_path):
            os.makedirs(self.scale_path)

        self.write_metadata(self.scale_path, '')
        self.write_metadata(self.scale_path, '1: MinMaxScaler')
        self.write_metadata(self.scale_path, '2: StandardScaler')
        self.write_metadata(self.scale_path, '3: RobustScaler')

        for i in range(7):
            for j in range(2):
                self.min_max_scaler(self.dfs[i].copy(), f'{i + 1}_{j + 1}')
                self.standard_scaler(self.dfs[i].copy(), f'{i + 1}_{j + 1}')
                self.df_robust_scaled(self.dfs[i].copy(), f'{i + 1}_{j + 1}')


    def run(self):
        self.get_df_list()
        self.scaler()