import numpy as np
from sklearn.ensemble import RandomForestRegressor

from models.base_model import BaseExtractionModel
from models.classic_model import ClassicModel


class MLModel(BaseExtractionModel):
    """
    ML-суррогат: Random Forest, обученный на синтетических данных
    (классическая модель + разброс D и шум измерений).
    Предсказывает долю металла, оставшуюся в водной фазе после ступени:
    x_i / feed. Признаки: число ступеней, O:A, номер ступени.
    """

    _rf = None  # обучается один раз на процесс

    def __init__(self):
        super().__init__("ML-модель (Random Forest, суррогат)")

    @classmethod
    def _train(cls, n_samples=1500, seed=42):
        rng = np.random.default_rng(seed)
        X, y = [], []
        for _ in range(n_samples):
            stages = int(rng.integers(1, 21))
            flow = rng.uniform(0.5, 5.0)
            D = max(rng.normal(0.6, 0.08), 0.05)  # разброс свойств системы
            conc, _ = ClassicModel(D).predict(stages, 1.0, flow)
            for i, c in enumerate(conc):
                noise = rng.normal(0, 0.02)  # шум "измерений"
                X.append([stages, flow, i])
                y.append(float(np.clip(c * (1 + noise), 0.0, 1.0)))
        cls._rf = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=-1
        ).fit(X, y)

    def predict(self, stages, feed_conc, flow_ratio):
        if MLModel._rf is None:
            MLModel._train()

        X = [[stages, flow_ratio, i] for i in range(stages + 1)]
        frac = MLModel._rf.predict(X)

        ml_conc = np.clip(frac * feed_conc, 0.0, feed_conc)
        ml_conc[0] = feed_conc

        recovery = (feed_conc - ml_conc[-1]) / feed_conc * 100
        return ml_conc, recovery
