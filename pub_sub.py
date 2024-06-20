import time
import zmq
import json

def start_publisher():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://127.0.0.1:5678")    
    print("Publisher started, publishing messages...")

    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://127.0.0.1:5679")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
    print("Subscriber started, publishing messages...")

    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)

    sequences = [
        ["w" for i in range(20)],
        ["l" for i in range(20)],
        ["j" for i in range(20)],
    ]
    
    messages = []
    
    for sequence in sequences:
        messages.extend(sequence)
    
    index = 0

    try:
        while True:
            message = messages[index]
            index += 1
            if index == len(messages):
                index = 0
                
            publisher.send_string(message)

            events = poller.poll(5)
            if events:
                message = subscriber.recv_string()
                message = json.loads(message)
                print(message)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Publisher interrupted, closing...")
    finally:
        publisher.close()
        context.term()

if __name__ == "__main__":
    start_publisher()
