"""
Mohammad A Moghaddam
Neural Network hw2
"""
import math
from typing import List

import numpy as np
import numpy.random as nprand

#sigmoid fun

def sigfun( x):
    return 1/(1+np.exp(-x))
# Derivative of Sigmoid

def deltasig(x):
    return sigfun( x) * (1-sigfun(x))    

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
        self.input_to_hidden_weights = input_to_hidden_weights
        self.hidden_to_output_weights = hidden_to_output_weights
        self.l1 = {}
        self.l2 = {}
        
        
    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
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
        #first layer 
        comb_lay1 = np.dot(input_matrix ,self.input_to_hidden_weights)
        self.l1['z'] = comb_lay1
        active_lay1  = sigfun(comb_lay1)
        self.l1['a'] = active_lay1
        
        #seond layer
        comb_lay2 = np.dot(active_lay1,self.hidden_to_output_weights)
        self.l2['z'] = comb_lay2
        active_lay2 = sigfun(comb_lay2 ) 
        self.l2['a'] = active_lay2

        
        return active_lay2
    
        
    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).
        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """
        
        #assert type(self.forwardprediction)==np.ndarray,' the forward prediction has not yet been done '
        zero_one = self.predict(input_matrix)>0.5 
        
        return 1*zero_one
    
        
    def gradients(self,
                  input_matrix: np.ndarray,
                  output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.
        For a single input example (where we write the input vector as o_{0}),
        this method:
        1. performs a pass of forward propagation through the network, keeping
           track of the weighted sums (before the activation function) of each
           unit (at layer l, we call such a vector z_{l}) and the activation
           (after the activation function) of each unit (at layer l, we call
           such a vector a_{l}).
        2. calculates the error of the final layer, using the output matrix
           (where y is the expected output vector) as:
               errors_{3} = a_{3} - y
        3. calculates the error of the hidden layer using the hidden-to-output
           matrix, W_{2}, and the sigmoid gradient, sigmoid'(z) =
           sigmoid(z)(1 - sigmoid(z)) as:
               errors_{2} = transpose(W_{2}) x errors_{3} * sigmoid'(z_{2})
           where sigmoid' is the sigmoid gradient,
        4. updates the gradients for both weight matrices as:
               gradient_{l} += errors_{l+1} x a_{l,i}
        When all input examples have applied their updates to the gradients,
        the entire gradient should be divided by the number of input examples.
        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: two matrices of gradients, one for the input-to-hidden weights
        and one for the hidden-to-output weights
        """
        
        # performs a pass of forward propagation through the network, keeping sum
          
        forward = self.predict(input_matrix)
        
        #calculates the error of the final layer
        
        error = forward - output_matrix
        delta_l2 = np.dot(self.l1['a'].T,error)
        
        #calculates the grad for the first layer
        
        error_h1 = np.dot(error, self.hidden_to_output_weights.T )
        error_h1 = np.multiply(error_h1, deltasig(self.l1['z'] ))
        delta_l1 = np.dot(input_matrix.T,error_h1)

        return delta_l1/input_matrix.shape[0],delta_l2/input_matrix.shape[0]
    
    
    def train(self,
              input_matrix: np.ndarray,
              output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.
        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.
        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """
        
        i=0
        while i<iterations+1:
          i += 1 
          delta_l1,delta_l2 = self.gradients(input_matrix,output_matrix)
          self.hidden_to_output_weights  -= learning_rate*delta_l2
          self.input_to_hidden_weights  -=learning_rate*delta_l1
          