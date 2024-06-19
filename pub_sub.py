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
    
    try:
        while True:
            message = input()
            publisher.send_string(message)
            # print(f"\nPublished: {message}")

            events = poller.poll(5)
            if events:
                message = subscriber.recv_string()
                message = message.split("SEP")
                transform_string = message[0].replace("Transform ", "")
                velocity_string = message[1].replace("Velocity ", "")
                # transform = json.loads(transform_string)
                # velocity = json.loads(velocity_string)
                print(transform_string)
                print(velocity_string)
    except KeyboardInterrupt:
        print("Publisher interrupted, closing...")
    finally:
        publisher.close()
        context.term()

if __name__ == "__main__":
    start_publisher()
