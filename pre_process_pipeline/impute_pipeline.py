import pandas as pd
import os

from sklearn.impute import SimpleImputer

class ImputePipeline:
    
    def __init__(self, config) -> None:
        self.encod_path = config['directory_encode_category']
        self.impute_path = config['directory_impute']
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
        directory = self.encod_path
        for i in range(7):
            dfs.append(pd.read_csv(f'{directory}/{i + 1}.csv'))

        self.dfs = dfs

    def impute_mean(self, df, df_name):
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        filepath = os.path.join(self.impute_path, f'{df_name}_1.csv')
        df_imputed.to_csv(filepath, index=False)

    def impute_median(self, df, df_name):
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        filepath = os.path.join(self.impute_path, f'{df_name}_2.csv')
        df_imputed.to_csv(filepath, index=False)

    def impute_numeric_cols(self):
        if not os.path.exists(self.impute_path):
            os.makedirs(self.impute_path)

        self.write_metadata(self.impute_path, '')
        self.write_metadata(self.impute_path, '1: impute (mean)')
        self.write_metadata(self.impute_path, '2: impute (median)')

        for i in range(len(self.dfs)):
            self.impute_mean(self.dfs[i].copy(), str(i + 1))
            self.impute_median(self.dfs[i].copy(), str(i + 1))


    def run(self):
        self.get_df_list()
        self.impute_numeric_cols()