from collections import deque
from typing import List, Deque, Any
import threading
import time
import zmq
import json
import copy


class SimulationInterface:
    def __init__(
        self,
        verbose=0,
        address="127.0.0.1",
        publisher_port="5678",
        subscriber_port="5679",
        poll_timeout_ms=5,
        frequency_hz=300,
        max_history_length=1800 # few minutes of playing
    ):
        self.address: str = address
        self.publisher_port: str = publisher_port
        self.subscriber_port: str = subscriber_port

        self.out_message: str = ""
        self.out_message_queue: Deque[str] = deque([])

        self.in_message: Dict[str, Any] = {}

        self.history: Deque[dict] = deque([])
        self.MAX_HISTORY_LENGTH = max_history_length

        self.score: int = 0
        self.scored: bool = False
        self.frozen_scored: bool = False
        
        self.frequency_hz: int = frequency_hz
        self.poll_timeout_ms: int = poll_timeout_ms
        self.sleep_time_s: float = round(1 / self.frequency_hz, 4)
        
        self._configure_zmq()


    def _configure_zmq(self):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.subscriber = self.context.socket(zmq.SUB)

        self.publisher.bind(f"tcp://{self.address}:{self.publisher_port}")
        self.subscriber.connect(f"tcp://{self.address}:{self.subscriber_port}")

        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

        self.poller = zmq.Poller()
        self.poller.register(self.subscriber, zmq.POLLIN)


    def _run(self):
        try:
            while True:
                self.scored = False
                self.in_message = {}
                self.out_message = ""

                self._subscribe()

                if not self.in_message:
                    continue
                
                if bool(self.out_message_queue):
                    self.out_message = self.out_message_queue.popleft()

                self._publish()
                self._update_history()
                
                time.sleep(self.sleep_time_s)
        except Exception as e:
            print(f"There was an error in the _run function: {e}")
        finally:
            self.context.destroy()


    def _publish(self):
        if self.out_message:
            self.publisher.send_string(self.out_message)


    def _subscribe(self):
        events = self.poller.poll(self.poll_timeout_ms)
        if events:
            self.in_message = json.loads(self.subscriber.recv_string())


    def _update_history(self):
        previous_score = copy.deepcopy(self.score)
        self.score = self.in_message["stars_eaten"]
        if self.score > previous_score:
            self.scored = True
            self.frozen_scored = True
            
        self.history.append(
            {
                "out": self.out_message,
                "in": self.in_message,
                "score": self.score,
                "scored": self.scored
            }
        )
        if len(self.history) > self.MAX_HISTORY_LENGTH:
            self.history.popleft()

        
    def run(self):
        threading.Thread(target=self._run).start()


    def add_message(self, message: str):
        self.out_message_queue.append(message)
        

    def add_message_with_priority(self, message: str):
        self.out_message_queue.appendleft(message)


    def add_messages(self, messages: List[str]):
        self.out_message_queue.extend(messages)        


    def add_messages_with_priority(self, messages: List[str]):
        self.out_message_queue.extendleft(messages.reverse())

    
    def set_messages(self, messages: List[str]):
        self.out_message_queue = deque(messages)


    def clear_messages(self):
        self.out_message_queue = deque([])


    def is_busy(self):
        return bool(self.out_message_queue)


    # TODO handle verbose level
