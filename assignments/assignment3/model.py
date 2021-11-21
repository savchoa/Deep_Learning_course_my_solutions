import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.conv1 = ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1)
        self.ReLu_1 = ReLULayer()
        self.Max_1 = MaxPoolingLayer(4, 2)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.ReLu_2 = ReLULayer()
        self.Max_2 = MaxPoolingLayer(3, 2)
        self.Flatten = Flattener()
        self.fc_layer = FullyConnectedLayer(98, n_output_classes)
        
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        self.conv1.params()['W'].grad = np.zeros_like(self.conv1.params()['W'].value)
        self.conv1.params()['B'].grad = np.zeros_like(self.conv1.params()['B'].value)
        self.conv2.params()['W'].grad = np.zeros_like(self.conv2.params()['W'].value)
        self.conv2.params()['B'].grad = np.zeros_like(self.conv2.params()['B'].value)
        self.fc_layer.params()['W'].grad = np.zeros_like(self.fc_layer.params()['W'].value)
        self.fc_layer.params()['B'].grad = np.zeros_like(self.fc_layer.params()['B'].value)
        
         
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        #         forward
        conv1_output = self.conv1.forward(X)
        ReLu_1_output = self.ReLu_1.forward(conv1_output)
        Max_1_output = self.Max_1.forward(ReLu_1_output)
        conv2_output = self.conv2.forward(Max_1_output)
        ReLu_2_output = self.ReLu_2.forward(conv2_output)
        Max_2_output = self.Max_2.forward(ReLu_2_output)
        Flatten_output = self.Flatten.forward(Max_2_output)
        fc_layer_output = self.fc_layer.forward(Flatten_output)
        loss_data, d_preds = softmax_with_cross_entropy(fc_layer_output, y)
        
        #         backward
        d_fc_layer = self.fc_layer.backward(d_preds)
        d_Flatten = self.Flatten.backward(d_fc_layer)
        d_Max_2 = self.Max_2.backward(d_Flatten)
        d_ReLu_2 = self.ReLu_2.backward(d_Max_2)
        d_conv2 = self.conv2.backward(d_ReLu_2)
        d_Max_1 = self.Max_1.backward(d_conv2)
        d_ReLu_1 = self.ReLu_1.backward(d_Max_1)
        d_conv1 = self.conv1.backward(d_ReLu_1)
                
        return loss_data

    
    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)
        #         forward
        conv1_output = self.conv1.forward(X)
        ReLu_1_output = self.ReLu_1.forward(conv1_output)
        Max_1_output = self.Max_1.forward(ReLu_1_output)
        conv2_output = self.conv2.forward(Max_1_output)
        ReLu_2_output = self.ReLu_2.forward(conv2_output)
        Max_2_output = self.Max_2.forward(ReLu_2_output)
        Flatten_output = self.Flatten.forward(Max_2_output)
        fc_layer_output = self.fc_layer.forward(Flatten_output)
        probs = softmax(fc_layer_output) 
        pred = np.argmax(probs, axis = 1)

        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        result['W1'] = self.conv1.params()['W']
        result['B1'] = self.conv1.params()['B']
        result['W2'] = self.conv2.params()['W']
        result['B2'] = self.conv2.params()['B']
        result['W3'] = self.fc_layer.params()['W']
        result['B3'] = self.fc_layer.params()['B']
        return result
