import time
from darcyai_engine.input.input_stream import InputStream
from darcyai_engine.stream_data import StreamData
from darcyai_engine.utils import timestamp

class SampleStream(InputStream):
    def __init__(self, max_runs=10):
        self.__stopped = True
        self.__max_runs = max_runs


    def stop(self):
        self.__stopped = True


    def stream(self):
        self.__stopped = False

        count = 0
        while not self.__stopped and count < self.__max_runs:
            count += 1
            time.sleep(1)

            yield(StreamData("Hello!", timestamp()))
