from typing import Deque, List, Dict, Any
from collections import deque
import json
from datetime import datetime


class Preprocessing:
    @staticmethod
    def trim_success(history: Deque[Dict]) -> List[Dict]:
        history.reverse()
        history = Preprocessing._trim_left(history)
        history.reverse()
        history = Preprocessing._trim_left(history)
        return list(history) 


    @staticmethod
    def _trim_left(history: Deque[Dict]):
        history_length = len(history)
        for index, element in enumerate(history):
            if element["scored"] and index != (history_length - 1):
                history = deque(list(history)[index:])
                break
        return history


    @staticmethod
    def save_sample(
        data: List[Dict],
        mode: str,
        directory="samples",
        filename_prefix: str="positive",
    ):
        now = datetime.now()
        filename = f"./{directory}/{filename_prefix}_{mode}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.json"
        with open(filename, 'w') as f:
            f.write(json.dumps(data, indent=4))
        print(f"Saved sample in file {filename}")
