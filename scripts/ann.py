from abc import ABC, abstractmethod
import numpy as np
import joblib
import json
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense, Input, Reshape, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

from scripts.database import DatabaseInterface
from scripts.machine_learning import MLClassifier


class RNNClassifier(MLClassifier):
    def __init__(
                self,
                mapping_path,
                mode,
                batch_size=32,
                epochs=100,
                validation_split=0.2,
                drop_empty_keys=False,
                database_path='./datasets/dataset.db',
                number_of_stars=10
        ):
        super().__init__(
            mapping_path,
            mode,
            drop_empty_keys,
            database_path,
            number_of_stars
        )
        self.n_last_samples = 1
        self.input_shape = (self.n_last_samples * (3 * number_of_stars + 1),1,)
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        
        self.scaler = MinMaxScaler()
        self.model = Sequential()
        self.number_of_classes = len(np.unique(list(self.mapping.values()))) + 2
        self._build()
        self._compile()
        
        self.history = None
        self.loss = None

        self.fit_callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
            tf.keras.callbacks.ModelCheckpoint(filepath='./models/model.{epoch:02d}-{val_loss:.2f}.keras')
        ]


    def _encode_targets(self, targets):
        encoded_targets = []
        for key in targets:
            encoded_targets.append(self.mapping[key])
        one_hot_encoded_targets = to_categorical(encoded_targets, num_classes=self.number_of_classes)
        return one_hot_encoded_targets


    def _decode_targets(self, targets):
        class_number = np.argmax(targets)
        return self.reverse_mapping[class_number]


    def _build(self):
        n = int(np.asarray(self.input_shape).flatten().sum())
        def LSTM_unit(scale=1, return_sequences=True):
          units = int(n * scale // 1)
          self.model.add(LSTM(units=units, return_sequences=return_sequences))
          self.model.add(BatchNormalization())
          self.model.add(Dropout(0.04))

        def GRU_unit(scale=1, return_sequences=True):
          units = int(n * scale // 1)
          self.model.add(GRU(units=units, return_sequences=return_sequences))
          self.model.add(BatchNormalization())
          self.model.add(Dropout(0.04))

        self.model.add(Input(self.input_shape))
        for scale in range(10):
            LSTM_unit(5)
        LSTM_unit(1, False)
        self.model.add(Dense(n))
        self.model.add(Dense(self.number_of_classes, activation='softmax'))

        self.model.summary()


    def _compile(self):
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.F1Score(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )


    def fit(
            self,
            epochs=None,
            batch_size=None,
            validation_split=None
        ):
        if epochs:
            self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size
        if validation_split:
            self.validation_split = validation_split
        print(f"Input shape: {self.input_shape}")
        scaled_X_train = self.scaler.fit_transform(self.X_train)
        self.history = self.model.fit(
            scaled_X_train,
            np.asarray(self.y_train),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=self.fit_callbacks
        )


    def evaluate(self):
        self.loss = self.model.evaluate(self.X_test, np.asarray(self.y_test))
        return self.loss


    def act(self, input, print_output=False):
        input = self.scaler.transform(input)
        encoded_prediction = self.model.predict(input)
        control_sequence = self._decode_targets(encoded_prediction[0])
        if print_output:
            print(control_sequence)
        return control_sequence


    def save(self, model_path):
        assert model_path.endswith('.keras')
        joblib.dump(self.scaler, './models/scaler.pkl')
        self.model.save(model_path)


    def load(self, model_path, scaler=False):
        assert model_path.endswith('.keras')
        if scaler:
            self.scaler = joblib.load('./models/scaler.pkl')
        self.model = tf.keras.models.load_model(model_path)
