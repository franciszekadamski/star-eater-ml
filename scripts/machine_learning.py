from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import os
import pandas as pd

from scripts.database import DatabaseInterface


class DecisionTreeActor():
    def __init__(
            self,
            dataset_dir,
            sample_dir,
            models_dir,
            model_path,
            mapping,
            mode,
            drop_empty_keys=False,
            model=None,
            operate_on_json=False,
            database_filename='dataset.db',
            number_of_stars=10
    ):
        self.dataset_dir = dataset_dir
        self.sample_dir = sample_dir
        self.models_dir = models_dir
        self.model_path = model_path
        self.mapping = mapping
        self.reverse_mapping = {number: key for key, number in mapping.items()}

        self.model = model
        if self.model:
            self.model_path = None

        if self.model_path:
            self._load_model()

        self.mode = mode
        self.drop_empty_keys = drop_empty_keys

        self.operate_on_json = operate_on_json
        
        self.n_last_samples = 56
        self.number_of_stars = number_of_stars
        
        self.data = {}
        self.X_data = []
        self.y_data = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        self.database = DatabaseInterface(os.path.join(self.dataset_dir, database_filename), self.n_last_samples * self.number_of_stars * 3 + self.n_last_samples)

    def _load_model(self):
        self.model = joblib.load(self.model_path)


    def save_model(self, filename):
        joblib.dump(self.model, os.path.join(self.models_dir, filename))


    def samples_to_database(self):
        self.database.remove()
        self.database.create()
        filenames = os.listdir(self.sample_dir)
        for file_number, filename in enumerate(filenames):
            if self.mode in filename:
                print(f"Processing file number {file_number}/{len(filenames)}")
                with open(os.path.join(self.sample_dir, filename), 'r') as f:
                    content = json.loads(f.read())
                X, y = self._accumulate_rows(content)
                self.database.insert(X, y)
                print(f"Processed file {filename}")


    def samples_to_dataset(self, output_filename):
        filenames = os.listdir(self.sample_dir)
        data = {
            "features": [],
            "targets": []
        }
        for file_number, filename in enumerate(filenames):
            if self.mode in filename:
                print(f"Processing file number {file_number}/{len(filenames)}")
                with open(os.path.join(self.sample_dir, filename), 'r') as f:
                    content = json.loads(f.read())
                X, y = self._accumulate_rows(content)
                data["features"].extend(X)
                data["targets"].extend(y)
                print(f"Processed file {filename}")

        assert data["features"] != []
        assert data["targets"] != []

        if self.operate_on_json:
            json_formatted = json.dumps(data, indent=4)
            with open(os.path.join(self.dataset_dir, output_filename), 'w') as f:
                f.write(json_formatted)
                print(f"Saved dataset {output_filename}")
        else:
            joblib.dump(data, os.path.join(self.dataset_dir, output_filename))               


    def _accumulate_rows(self, sample):
        input_X = sample["observations"]
        input_y = sample["actions"]
        
        output_X = []
        output_y = []
        
        for _ in range(len(input_X) - self.n_last_samples):
            y = input_y.pop()
            X = self._adjust_input(input_X)
            input_X.pop()

            if not self.drop_empty_keys:
                output_X.append(X)
                output_y.append(y)
            else:
                if y != "":
                    output_X.append(X)
                    output_y.append(y)
        
        return output_X, output_y
            

    def read_dataset(self, filepath):
        if filepath.endswith('.db'):
            data_df = self.database.get_data()
            row_length = len(data_df.iloc[0]) - 2
            self.X_data = data_df[[f'feature_{i}' for i in range(row_length)]].to_numpy()
            self.y_data = data_df['label'].to_numpy()
        else:
            if self.operate_on_json:
                with open(os.path.join(self.dataset_dir, filepath), 'r') as f:
                    data = json.loads(f.read())
                    self.X_data = data["features"]
                    self.y_data = data["targets"]
            else:
                data = joblib.load(self.dataset_dir, filepath)
                self.X_data = data["features"]
                self.y_data = data["targets"]
            

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
    

    def train(self):
        print("Training model")
        self.model.fit(self.X_train, self.y_train)
        print("Finished training")

    def partial_train(self, data):
        pass


    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)
        

    def _adjust_input(self, input):
        input = np.asarray(input[-self.n_last_samples:]).flatten()
        return list(input)


    def act(self, input):
        try:
            assert type(input) == list
        except AssertionError:
            raise Exception("Act method should receive list of observations. Are you sure you are passing only observation list instead of a dictionary? \nHint: example of usage: \n\tActorChild.act(data['obseravtions'])")
            
        input = self._adjust_input(input)
        encoded_prediction = self.model.predict([input])
        control_sequence = self._decode_targets(encoded_prediction)
        print(control_sequence)
        return control_sequence


    def dummy_act(self, input):
        return ['w' for _ in range(10)]
