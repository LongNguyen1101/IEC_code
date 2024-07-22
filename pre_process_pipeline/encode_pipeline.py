import pandas as pd
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from category_encoders import BinaryEncoder, TargetEncoder, LeaveOneOutEncoder

class EncodePipeline: 
    
    def __init__(self, config) -> None:
        self.config = config
        data = pd.read_csv(self.config['data_path'])

        self.data = data
        self.shape = data.shape
        self.category_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        self.X = None
        self.y = None
        self.metadata_filename = config['metadata']

    def drop_null_cols(self):
        self.data.dropna(axis=1, how='all', inplace=True)

    def drop_duplicate(self):
        self.data.drop_duplicates(inplace=True)

    def impute_category_cols(self):
        imputer = SimpleImputer(strategy='most_frequent')
        self.data[self.category_cols] = imputer.fit_transform(self.data[self.category_cols])

    def write_metadata(self, directory, content):
        filetxt = os.path.join(directory, self.metadata_filename)
        if content == '':
          with open(filetxt, 'w') as file:
            file.write(content)
        else:
            with open(filetxt, 'a') as file:
                file.write(content + '\n')

    def label_encoding(self, directory, filename):
        data = self.data.copy()
        label_encoder = LabelEncoder()
        for col in self.category_cols:
            data[col] = label_encoder.fit_transform(data[col])

        filepath = os.path.join(directory, filename)
        data.to_csv(filepath, index=False)

        self.write_metadata(directory, '1: label_encoding')

    def one_hot_encoding(self, directory, filename):
        data = pd.get_dummies(self.X, columns=self.category_cols, dtype=float)

        data['label'] = self.y

        filepath = os.path.join(directory, filename)
        data.to_csv(filepath, index=False)

        self.write_metadata(directory, '2: one_hot_encoding')

    def binary_encoding(self, directory, filename):
        data = self.X.copy()
        binary_encoder = BinaryEncoder()
        data_binary = binary_encoder.fit_transform(data[self.category_cols])

        data = data.drop(self.category_cols, axis=1)

        data = pd.concat([data, data_binary, self.y], axis=1)

        filepath = os.path.join(directory, filename)
        data.to_csv(filepath, index=False)

        self.write_metadata(directory, '3: binary_encoding')

    def ordinal_encoding(self, directory, filename):
        data = self.data.copy()
        ordinal_encoder = OrdinalEncoder()
        data[self.category_cols] = ordinal_encoder.fit_transform(data[self.category_cols])

        filepath = os.path.join(directory, filename)
        data.to_csv(filepath, index=False)

        self.write_metadata(directory, '4: ordinal_encoding')

    def frequency_encoding(self, directory, filename):
        data = self.data.copy()
        frequency_encodings = []

        for col in self.category_cols:
            frequency_encodings.append(data[col].value_counts(normalize=True))
        
        for i in range(len(self.category_cols)):
            data[self.category_cols[i]] = data[self.category_cols[i]].map(frequency_encodings[i])

        filepath = os.path.join(directory, filename)
        data.to_csv(filepath, index=False)

        self.write_metadata(directory, '5: frequency_encoding')

    def target_encoder(self, directory, filename):
        data = self.data.copy()

        target_encoder = TargetEncoder()
        data[self.category_cols] = target_encoder.fit_transform(data[self.category_cols], data['label'])

        filepath = os.path.join(directory, filename)
        data.to_csv(filepath, index=False)

        self.write_metadata(directory, '6: target_encoder')

    def loo_encoder(self, directory, filename):
        data = self.data.copy()

        loo_encoder = LeaveOneOutEncoder()
        data[self.category_cols] = loo_encoder.fit_transform(data[self.category_cols], data['label'])

        filepath = os.path.join(directory, filename)
        data.to_csv(filepath, index=False)

        self.write_metadata(directory, '7: loo_encoder')


    def encoding_category_cols(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.write_metadata(directory, '')

        self.label_encoding(directory, '1.csv')
        self.one_hot_encoding(directory, '2.csv')
        self.binary_encoding(directory, '3.csv')
        self.ordinal_encoding(directory, '4.csv')
        self.frequency_encoding(directory, '5.csv')
        self.target_encoder(directory, '6.csv')
        self.loo_encoder(directory, '7.csv')


    def run(self):
        self.drop_null_cols()
        if self.config['drop_dup'] == True: 
            self.drop_duplicate()
        self.impute_category_cols()

        self.X = self.data.drop('label', axis=1)
        self.y = self.data['label']

        self.encoding_category_cols(self.config['directory_encode_category'])