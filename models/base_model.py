class BaseExtractionModel:
    def __init__(self, name):
        self.name = name

    def predict(self, stages, feed_conc, flow_ratio):
        raise NotImplementedError("Метод predict должен быть реализован в дочернем классе")