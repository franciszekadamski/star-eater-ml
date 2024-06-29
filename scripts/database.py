#!/bin/python3

import sqlite3
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import random

class DatabaseInterface:
    def __init__(self, filename, number_of_features):
        self.number_of_features = number_of_features
        self.filename = filename


    def _connect(self):
        self.connection = sqlite3.connect(self.filename)
        self.cursor = self.connection.cursor()


    def _disconnect(self):
        self.connection.commit()
        self.connection.close()        


    def create(self):
        with open(self.filename, 'w') as f:
            f.write('')
            
        self._connect()
        columns = ', '.join([f'feature_{i} REAL' for i in range(self.number_of_features)]) + ', label CHAR'
        self.cursor.execute(f'CREATE TABLE IF NOT EXISTS data (id INTEGER PRIMARY KEY, {columns})')
        self._disconnect()


    def remove(self):
        try:
            os.remove(self.filename)
        except:
            print("Could not remove database file")

    def insert(self, features, targets):
        self._connect()
        query = 'INSERT INTO data ({}) VALUES ({})'.format(
            ', '.join([f'feature_{i}' for i in range(self.number_of_features)] + ['label']),
            ', '.join(['?'] * (self.number_of_features + 1))
        )

        for i in range(len(features)):
            row = tuple(features[i]) + (targets[i],)
            self.cursor.execute(query, row)
            
        self._disconnect()


    def get_data(self):
        self._connect()
        query = 'SELECT * FROM data WHERE id >= 0'
        df = pd.read_sql_query(query, self.connection)
        self._disconnect()
        return df


    def get_row_range(self, start, end):
        self._connect()
        query = f'SELECT * FROM data WHERE id >= {start} AND id <= {end - 1}'
        df = pd.read_sql_query(query, self.connection)
        self._disconnect()
        return df


def test():
    number_of_features = 10
    number_of_rows = 1000
    possible_targets = [0, 1, 2]
    
    database = DatabaseInterface("test/dataset.db", number_of_features)
    database.create()
    chunk = 1000
    for _ in range(int(number_of_rows / chunk)):
        features = np.random.rand(chunk, number_of_features)
        targets = [random.choice(possible_targets) for _ in range(chunk)]
        database.insert(features, targets)
    slice = database.get_row_range(500, 510)
    # print(slice)
    # database.remove()
    
        
if __name__ == "__main__":
    test()
