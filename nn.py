import math
from typing import List

import numpy as np
import numpy.random as nprand


class SimpleNetwork:
    """A simple feedforward network with a single hidden layer. All units in
    the network have sigmoid activation.

    """

    @classmethod
    def of(cls, n_input: int, n_hidden: int, n_output: int):
        """Creates a single-layer feedforward neural network with the given
        number of input, hidden, and output units.

        :param n_input: Number of input units
        :param n_hidden: Number of hidden units
        :param n_output: Number of output units
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return nprand.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        return cls(uniform(n_input, n_hidden), uniform(n_hidden, n_output))

    def __init__(self,
                 input_to_hidden_weights: np.ndarray,
                 hidden_to_output_weights: np.ndarray):
        """Creates a neural network from two weights matrices, one representing
        the weights from input units to hidden units, and the other representing
        the weights from hidden units to output units.

        :param input_to_hidden_weights: The weight matrix mapping from input
        units to hidden units
        :param hidden_to_output_weights: The weight matrix mapping from hiddden
        units to output units
        """

    def predict(self, input_matrix) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a logistic sigmoid activation function.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each in the range (0, 1) - for the corresponding row in the
        input matrix.
        """

    def predict_zero_one(self, input_matrix) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """

    def gradients(self, input_matrix, output_matrix) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.

        For a single input example (where we write each feature of the input as
        o_{i, 0}), this method:
        1. performs a pass of forward propagation through the network, keeping
           track of the activations of each unit (a_{i,l} is the activation of
           unit i in layer l) and the output of each unit (o_{i,l} is the output
           of unit i in layer l).
        2. calculates the error of the final layer, using the output matrix
           (where y_{i} is the ith element of the output) as
           errors_{i,3} = o_{i,3} - y_{i}
        3. updates the gradient for the hidden-to-output matrix as
           gradient_{i,2} += errors_{i, 3} x a_{i,2}
        4. calculates the error of the hidden layer as
           errors_{1,2}


        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: two matrices of gradients, one for the input-to-hidden weights
        and one for the hidden-to-output weights
        """

    def train(self,
              input_matrix,
              output_matrix,
              iterations=1000,
              learning_rate=0.1):
        pass
