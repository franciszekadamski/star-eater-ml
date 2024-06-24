from collections import deque
from typing import Deque, List, Any
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import sys
import termios
import tty
import select
import copy
import json

from scripts.process_controller import ProcessController
from scripts.simulation_interface import SimulationInterface
from scripts.preprocessing import Preprocessing
from scripts.observation_transformer import ObservationTransformer
from scripts.machine_learning import DecisionTreeActor


class Bootstrap:
    def __init__(
            self,
            mode="mlmodel",
            save_scored=False,
            verbose=0,
            transform_single_mode="MIN",
            ml_data_dir="./datasets",
            sample_dir="./samples",
            models_dir="./models",
            model_path="./models/model.pkl",
            mapping_path="./mapping.json"
    ):
        self.mode = mode
        self.save_scored = save_scored
        self.verbose = verbose
        self.transform_single_mode = transform_single_mode
        self.ml_data_dir = ml_data_dir
        self.sample_dir = sample_dir
        self.models_dir = models_dir
        self.model_path = model_path
        self.mapping = self._load_mapping(mapping_path)

        self.process_controller = ProcessController(verbose=self.verbose, flags=['--release'])
        self.simulation = SimulationInterface(verbose=self.verbose)
        self.observation_transformer = ObservationTransformer(["MIN", "ALL"])
        self.actor = DecisionTreeActor(
            dataset_dir=self.ml_data_dir,
            sample_dir=self.sample_dir,
            models_dir=self.models_dir,
            model_path=self.model_path,
            mapping=self.mapping,
            mode=self.transform_single_mode
        )

        self.process_controller.start_simulation()
        time.sleep(5)
        self.simulation.run()

        self.sequence = []
        self.sequences = []

        self.allowed_keys = list(self.mapping.keys())[1:]

        self.thread_executor = ThreadPoolExecutor(max_workers=10)
        self.lock = threading.Lock()
        self.task_queue = Queue()


    def _load_mapping(self, mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.loads(f.read())
        return mapping

        
    def run(self):
        time.sleep(1)
        if self.mode == "user":
            while True:
                key = self._read_single_keypress()
                if key:
                    self.simulation.add_messages(key)
                    self._handle_just_scored()
                    self._wait_until_finished()
        elif self.mode == "sequence":
            while True:
                for sequence in self.sequences:
                    self.simulation.add_messages(sequence)
                    self._handle_just_scored()
                    self._wait_until_finished()
        elif self.mode == "mlmodel":
            while True:
                data = ObservationTransformer.transform_single(
                    input_sample=self.simulation.history,
                    mode=self.transform_single_mode,
                    trim=3
                )
                self.sequences = self.actor.act(data["observations"])
                for sequence in self.sequences:
                    self.simulation.add_messages(sequence)
                    self._handle_just_scored()
                    self._wait_until_finished()
        else:
            raise Exception("There is no such mode in Bootstrap class")


    def _read_single_keypress(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    
        try:
            tty.setcbreak(fd)
            if select.select([sys.stdin], [], [], 0.001)[0]:
                key = sys.stdin.read(1)
                if key in self.allowed_keys:
                    return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
        return None

    
    def _handle_just_scored(self):
        if self.simulation.frozen_scored and self.save_scored:
            self.simulation.frozen_scored = False
            self.task_queue.put(self.simulation.history)
            self.thread_executor.submit(self._process_task)


    def _process_task(self):
        while True:
            try:
                history = self.task_queue.get(timeout=5)
                self._trim_transform_and_save(history)
                self.task_queue.task_done()
            except Emtpy:
                break
            

    def _trim_transform_and_save(self, history):
        with self.lock:
            sample = Preprocessing.trim_success(history)
            samples = self.observation_transformer.transform(sample)
            for mode, sample in zip(self.observation_transformer.modes, samples):
                Preprocessing.save_sample(data=sample, mode=mode)

                
    def _wait_until_finished(self, frequency_hz=120):
        while self.simulation.is_busy():
            time.sleep(1 / frequency_hz)

    
    def use_dummy_sequence(self, n=5):
        self.sequences = [
            ['w' for _ in range(n)]
        ]
        print("Setting mode to 'sequence'")
        self.mode = "sequence"
