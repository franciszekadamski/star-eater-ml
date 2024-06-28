from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib
import json
import os
import pandas as pd

from scripts.database import DatabaseInterface


class MLClassifier():
    def __init__(
            self,
            mapping_path,
            mode,
            drop_empty_keys=False,
            database_path='./datasets/dataset.db',
            number_of_stars=10
    ):
        self._read_mapping(mapping_path)
        self.n_last_samples = 1
        self.model = KNeighborsClassifier()
        self.database = DatabaseInterface(
            database_path,
            self.n_last_samples * (3*number_of_stars + 1)
        )
        self.mode = mode
        self.drop_empty_keys = drop_empty_keys
        self.data = {}
        self.X_data = []
        self.y_data = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []


    def _read_mapping(self, path):
        with open(path, 'r') as f:
            self.mapping = json.loads(f.read())
        self.reverse_mapping = {number: key for key, number in self.mapping.items()}


    def load(self, model_path):
        self.model = joblib.load(model_path)


    def save(self, model_path):
        joblib.dump(self.model, model_path)


    def samples_to_database(self):
        self.database.remove()
        self.database.create()
        filenames = os.listdir("./samples")
        for file_number, filename in enumerate(filenames):
            if self.mode in filename:
                print(f"Processing file number {file_number}/{len(filenames)}")
                with open(os.path.join('./samples', filename), 'r') as f:
                    content = json.loads(f.read())
                X, y = self._accumulate_rows(content)
                self.database.insert(X, y)
                print(f"Processed file {filename}")


    def _accumulate_rows(self, sample):
        input_X = sample["observations"]
        input_y = sample["actions"]        
        output_X = []
        output_y = []
        for _ in range(len(input_X) - self.n_last_samples):
            y = input_y.pop()
            X = self._get_n_last_rows(input_X, self.n_last_samples)
            input_X.pop()
            if not self.drop_empty_keys:
                output_X.append(X)
                output_y.append(y)
            else:
                if y != "":
                    output_X.append(X)
                    output_y.append(y)
        return output_X, output_y


    def read_dataset(self):
        data_df = self.database.get_data()
        row_length = len(data_df.iloc[0]) - 2
        self.X_data = data_df[[f'feature_{i}' for i in range(row_length)]].to_numpy()
        self.y_data = data_df['label'].to_numpy()            


    def train_test_split(self, validation=False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_data,
            self.y_data,
            test_size=0.3
        )
        self.y_train = self._encode_targets(self.y_train)
        self.y_test = self._encode_targets(self.y_test)


    def _encode_targets(self, targets):
        encoded_targets = []
        for key in targets:
            encoded_targets.append(self.mapping[key])
        return encoded_targets


    def _decode_targets(self, targets):
        decoded_targets = []
        for class_number in targets:
            decoded_targets.append(self.reverse_mapping[class_number])
        return decoded_targets


    def fit(self):
        self.model.fit(self.X_train, self.y_train)


    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
        

    def _get_n_last_rows(self, input, n_last):
        input = np.asarray(input[-n_last:]).flatten()
        return list(input)


    def act(self, input, print_output=False):
        try:
            assert type(input) == list
        except AssertionError:
            raise Exception("Act method should receive list of observations. Are you sure you are passing only observation list instead of a dictionary? \nHint: example of usage: \n\tActorChild.act(data['obseravtions'])")
            
        input = self._get_n_last_rows(input, self.n_last_samples)
        encoded_prediction = self.model.predict([input])
        control_sequence = self._decode_targets(encoded_prediction)
        if print_output:
            print(control_sequence)
        return control_sequence
