import numpy as np

from models.base_model import BaseExtractionModel
from models.classic_model import ClassicModel
from models.ml_model import MLModel


class HybridModel(BaseExtractionModel):
    """
    Гибрид: физика задаёт основу, ML поправляет остаток.
    conc = classic + w * (ml - classic), где w - вес ML-коррекции.
    """

    def __init__(self, ml_weight=0.5):
        super().__init__("Гибридная (физика + ML-поправка)")
        self.w = ml_weight
        self._classic = ClassicModel()
        self._ml = MLModel()

    def predict(self, stages, feed_conc, flow_ratio):
        classic_conc, _ = self._classic.predict(stages, feed_conc, flow_ratio)
        ml_conc, _ = self._ml.predict(stages, feed_conc, flow_ratio)

        classic_conc = np.array(classic_conc)
        hybrid_conc = classic_conc + self.w * (np.array(ml_conc) - classic_conc)
        hybrid_conc = np.clip(hybrid_conc, 0.0, feed_conc)

        recovery = (feed_conc - hybrid_conc[-1]) / feed_conc * 100
        return hybrid_conc, recovery
