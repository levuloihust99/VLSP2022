import math


class Statistics(object):
    """Accumulator for metrics."""

    def __init__(self, loss: float = 0.0, n_tokens: int = 0, n_corrects: int = 0):
        self.loss = loss
        self.n_tokens = n_tokens
        self.n_corrects = n_corrects

    def update(self, other: "Statistics"):
        self.loss += other.loss
        self.n_tokens += other.n_tokens
        self.n_corrects += other.n_corrects

    def ppl(self):
        return math.exp(self.loss / self.n_tokens)

    def acc(self):
        return self.n_corrects / self.n_tokens
