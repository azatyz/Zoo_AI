from models.base_model import BaseExtractionModel


class ClassicModel(BaseExtractionModel):
    def __init__(self):
        super().__init__("Классическая (Эмпирическая)")

    def predict(self, stages, feed_conc, flow_ratio):
        efficiency = 0.15 * flow_ratio
        concentrations = [feed_conc]

        for i in range(1, stages + 1):
            next_conc = concentrations[-1] * (1 - efficiency)
            concentrations.append(max(next_conc, 0.01))

        recovery = (
            (feed_conc - concentrations[-1]) / feed_conc * 100
        )

        return concentrations, recovery