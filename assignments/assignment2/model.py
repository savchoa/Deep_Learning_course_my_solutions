import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax 


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.fc_layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLu_1 = ReLULayer()
        self.fc_layer2 = FullyConnectedLayer(hidden_layer_size, n_output)
        
        
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        self.fc_layer1.params()['W'].grad = np.zeros_like(self.fc_layer1.params()['W'].value)
        self.fc_layer1.params()['B'].grad = np.zeros_like(self.fc_layer1.params()['B'].value)
        self.fc_layer2.params()['W'].grad = np.zeros_like(self.fc_layer2.params()['W'].value)
        self.fc_layer2.params()['B'].grad = np.zeros_like(self.fc_layer2.params()['B'].value)
         
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        #         forward
        fc_layer1_output = self.fc_layer1.forward(X)
        ReLu_1_output = self.ReLu_1.forward(fc_layer1_output)
        fc_layer2_output = self.fc_layer2.forward(ReLu_1_output)
        loss_data, d_preds = softmax_with_cross_entropy(fc_layer2_output, y)
        
        #         backward
        d_fc_layer2 = self.fc_layer2.backward(d_preds)
        d_ReLu_1 = self.ReLu_1.backward(d_fc_layer2)
        d_fc_layer1 = self.fc_layer1.backward(d_ReLu_1)
#         print(self.fc_layer1.params()['W'].grad.shape)
#         print(self.fc_layer1.params()['B'].grad.shape)
#         print(self.fc_layer2.params()['W'].grad.shape)
#         print(self.fc_layer2.params()['B'].grad.shape)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        loss_reg = l2_regularization(self.fc_layer1.params()['W'].value, self.reg)[0] + l2_regularization(self.fc_layer2.params()['W'].value, self.reg)[0]
        grad_reg_fc_layer1 = l2_regularization(self.fc_layer1.params()['W'].value, self.reg)[1] 
        grad_reg_fc_layer2 = l2_regularization(self.fc_layer2.params()['W'].value, self.reg)[1]
        self.fc_layer1.params()['W'].grad += grad_reg_fc_layer1
        self.fc_layer2.params()['W'].grad += grad_reg_fc_layer2
        
        loss = loss_data + loss_reg 
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        #         forward
        fc_layer1_output = self.fc_layer1.forward(X)
        ReLu_1_output = self.ReLu_1.forward(fc_layer1_output)
        fc_layer2_output = self.fc_layer2.forward(ReLu_1_output)
        probs = softmax(fc_layer2_output)
        pred = np.argmax(probs, axis = 1)

        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        result['W1'] = self.fc_layer1.params()['W']
        result['B1'] = self.fc_layer1.params()['B']
        result['W2'] = self.fc_layer2.params()['W']
        result['B2'] = self.fc_layer2.params()['B']

        return result
