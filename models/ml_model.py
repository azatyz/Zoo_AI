import numpy as np

from models.base_model import BaseExtractionModel
from models.classic_model import ClassicModel


class MLModel(BaseExtractionModel):
    def __init__(self):
        super().__init__("ML-модель (Random Forest - Заглушка)")

    def predict(self, stages, feed_conc, flow_ratio):
        base_model = ClassicModel()
        base_conc, _ = base_model.predict(
            stages, feed_conc, flow_ratio
        )

        noise = np.random.normal(
            0, feed_conc * 0.02, stages + 1
        )

        ml_conc = np.clip(
            np.array(base_conc) + noise,
            0.01,
            feed_conc
        )

        ml_conc[0] = feed_conc

        recovery = (
            (feed_conc - ml_conc[-1]) / feed_conc * 100
        )

        return ml_conc, recovery