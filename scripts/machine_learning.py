from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib
import json
import os


class DecisionTreeActor():
    def __init__(
            self,
            dataset_dir,
            sample_dir,
            models_dir,
            model_path,
            mapping
    ):
        self.dataset_dir = dataset_dir
        self.sample_dir = sample_dir
        self.models_dir = models_dir
        self.model_path = model_path
        self.mapping = mapping
        self.reverse_mapping = {number: key for key, number in mapping.items()}
        self.model = DecisionTreeClassifier()
        if self.model_path:
            self._load_model()

        self.n_last_samples = 3

        self.data = {}
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        

    def _load_model(self):
        self.model = joblib.load(self.model_path)


    def save_model(self, filename):
        joblib.dump(self.model, filename)


    def samples_to_dataset(self, output_filename):
        filenames = os.listdir(self.sample_dir)
        for filename in filenames:
            with open(filename, 'r') as f:
                content = json.loads(f.read())                
            print(f"{filename}:\n{content}")


    def read_dataset(self, filepath):
        with open(filepath, 'r') as f:
            self.data = json.loads(f.read())


    def train_test_split(self, validation=False):
        self._adjust_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data["observations"],
            self.data["actions"],
            test_size=0.3
        )
        self.y_train = self._encode_targets(self.y_train)
        self.y_test = self._encode_targest(self.y_test)


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
        self.model.fit(self.X_train, self.y_train)


    def partial_train(self, data):
        pass


    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        

    def _adjust_input(self, input):
        input = np.asarray(input[-self.n_last_samples:]).flatten()
        return input


    def act(self, input):
        try:
            assert type(input) == list
        except AssertionError:
            raise Exception("Act method should receive list of observations. Are you sure you are passing only observation list instead of a dictionary? \nHint: example of usage: \n\tActorChild.act(data['obseravtions'])")
            
        input = self._adjust_input(input)
        encoded_prediction = self.model.predict(input)
        control_sequence = self._decode_targets(encoded_prediction)
        return control_sequence


    def dummy_act(self, input):
        return ['w' for _ in range(10)]

