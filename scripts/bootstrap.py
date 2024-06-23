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

from scripts.process_controller import ProcessController
from scripts.simulation_interface import SimulationInterface
from scripts.preprocessing import Preprocessing
from scripts.observation_transformer import ObservationTransformer
from scripts.machine_learning_models import Actor


class Bootstrap:
    def __init__(
            self, mode="mlmodel",
            save_scored=False,
            verbose=0,
            transform_single_mode="MIN"
    ):
        self.mode = mode
        self.save_scored = save_scored
        self.verbose = verbose
        self.transform_single_mode = transform_single_mode
        
        self.process_controller = ProcessController(verbose=self.verbose, flags=['--release'])
        self.simulation = SimulationInterface(verbose=self.verbose)
        self.observation_transformer = ObservationTransformer(["MIN", "ALL"])
        self.actor = Actor()

        self.process_controller.start_simulation()
        self.simulation.run()

        self.sequence = []
        self.sequences = []

        self.allowed_keys = ['w', 'a', 's', 'd', 'q', 'e', 'j', 'k', 'l', 'i']

        self.thread_executor = ThreadPoolExecutor(max_workers=10)
        self.lock = threading.Lock()
        self.task_queue = Queue()


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
                observations = ObservationTransformer.transform_single(
                    input_sample=self.simulation.history,
                    mode=self.transform_single_mode,
                    trim=3
                )
                self.sequences = self.actor.act(observations)
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
