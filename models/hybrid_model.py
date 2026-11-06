import numpy as np

from models.base_model import BaseExtractionModel
from models.classic_model import ClassicModel
from models.ml_model import MLModel


class HybridModel(BaseExtractionModel):
    def __init__(self):
        super().__init__("Гибридная (Blending Classic + ML)")

    def predict(self, stages, feed_conc, flow_ratio):
        classic_conc, _ = ClassicModel().predict(
            stages, feed_conc, flow_ratio
        )

        ml_conc, _ = MLModel().predict(
            stages, feed_conc, flow_ratio
        )

        hybrid_conc = (
            np.array(classic_conc) + np.array(ml_conc)
        ) / 2

        recovery = (
            (feed_conc - hybrid_conc[-1]) / feed_conc * 100
        )

        return hybrid_conc, recovery