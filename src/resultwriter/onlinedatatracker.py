from typing import Tuple


class OnlineDataTracker:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_aggr(self, new_val: float) -> None:
        self.count += 1
        delta = new_val - self.mean
        self.mean += delta / self.count
        delta2 = new_val - self.mean
        self.M2 += delta * delta2

    def get_curr_mean_variance(self) -> Tuple[float, float]:
        mean, _, var = self._finalize_aggr()
        return mean, var

    def _finalize_aggr(self) -> Tuple[float, float, float]:
        if self.count < 2:
            return self.mean, 0, 0
        else:
            variance = self.M2 / self.count
            sample_variance = self.M2 / (self.count - 1)
            return self.mean, variance, sample_variance

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
