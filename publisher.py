import time
import zmq

def start_publisher():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://127.0.0.1:5678")

    print("Publisher started, publishing messages...")
    try:
        while True:
            message = input("Input a control sequence:\n")
            publisher.send_string(message)
            print(f"Published: {message}")
    except KeyboardInterrupt:
        print("Publisher interrupted, closing...")
    finally:
        publisher.close()
        context.term()

if __name__ == "__main__":
    start_publisher()
