from abc import ABC, abstractmethod

class BaseExtractionModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def predict(self, stages, feed_conc, flow_ratio):
        pass