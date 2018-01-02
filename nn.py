import math
from typing import List

import numpy as np
import numpy.random as nprand


class SimpleNetwork:
    @classmethod
    def of(cls, n_input, n_hidden, n_output):

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return nprand.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        return cls(uniform(n_input, n_hidden), uniform(n_hidden, n_output))

    def __init__(self, input_to_hidden_weights, hidden_to_output_weights):
        pass

    def predict(self, input_matrix) -> np.ndarray:
        pass

    def predict_zero_one(self, input_matrix) -> np.ndarray:
        pass

    def gradients(self, input_matrix, output_matrix) -> List[np.ndarray]:
        pass

    def train(self,
              input_matrix,
              output_matrix,
              iterations=1000,
              learning_rate=0.1):
        pass
