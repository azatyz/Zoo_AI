import numpy as np

from models.base_model import BaseExtractionModel


class ClassicModel(BaseExtractionModel):
    """
    Противоточный экстракционный каскад с линейной изотермой y = D * x.

    Водная фаза (поток A) с концентрацией feed_conc входит в ступень 1,
    чистый экстрагент (поток O) входит в ступень N. На каждой ступени
    достигается равновесие, поэтому баланс по ступеням - линейная система.
    Результат совпадает с формулой Кремзера.
    """

    def __init__(self, distribution_coeff=0.6):
        super().__init__("Классическая (противоточный каскад, равновесные ступени)")
        self.D = distribution_coeff  # коэффициент распределения

    def predict(self, stages, feed_conc, flow_ratio):
        N = stages
        E = self.D * flow_ratio  # фактор экстракции

        # Баланс ступени i (A = 1, O = flow_ratio, y_i = D * x_i):
        # x_{i-1} + E * x_{i+1} - (1 + E) * x_i = 0, x_0 = feed, x_{N+1} -> y_{N+1} = 0
        M = np.zeros((N, N))
        b = np.zeros(N)
        for i in range(N):
            M[i, i] = -(1 + E)
            if i > 0:
                M[i, i - 1] = 1
            if i < N - 1:
                M[i, i + 1] = E
        b[0] = -feed_conc

        x = np.linalg.solve(M, b)
        concentrations = [feed_conc] + [max(float(v), 0.0) for v in x]

        recovery = (feed_conc - concentrations[-1]) / feed_conc * 100
        return concentrations, recovery
