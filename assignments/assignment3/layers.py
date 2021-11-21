import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    if np.ndim(predictions)==1:
        pred = predictions - np.max(predictions)
        probs = np.exp(pred)/np.sum(np.exp(pred))
    
    else:
        pred = predictions - np.max(predictions, axis=1).reshape(-1, 1)
        probs = np.exp(pred)/(np.sum(np.exp(pred), axis=1).reshape(-1, 1))
    
    return probs

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if np.ndim(probs)==1:
        loss = - np.log(probs[target_index])
    else:
        loss = - np.log(probs[np.arange(probs.shape[0]), target_index.flatten()])
        loss = np.mean(loss)
        
    return loss

def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    d_preds = probs
    
    if np.ndim(preds)==1:
        d_preds[target_index] -= 1
    
    else:
        d_preds[np.arange(d_preds.shape[0]), target_index.flatten()] -= 1
        d_preds = d_preds/d_preds.shape[0]

    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        output = X.copy()
        self.X = X.copy()
        output[output<0] = 0
        return output
        
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_ReLU = np.ones(d_out.shape)
        d_ReLU[self.X<0] = 0 
        d_result = d_out*d_ReLU
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.1 * np.random.randn(n_input, n_output))
 
        self.B = Param(0.1 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        output = np.dot(X,self.W.value)+self.B.value
        return output

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        
        d_input = np.dot(d_out, self.W.value.T)
        self.W.grad = np.dot(self.X.T,d_out)
        self.B.grad = np.sum(d_out, axis=0).reshape(1, -1)
        
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = int(((height-self.filter_size+2*self.padding)/1)+1)
        out_width = int(((width-self.filter_size+2*self.padding)/1)+1)
        Conv_out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        self.X = X.copy()
        X_pad = np.pad(self.X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        W_resh=self.W.value.reshape(-1, self.W.value.shape[-1])
        for y in range(out_height): 
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                X_pad_view=X_pad[:,y:(y+self.filter_size),x:(x+self.filter_size),:]
                X_pad_view_resh = X_pad_view.reshape(X_pad_view.shape[0],-1)
                c=np.dot(X_pad_view_resh, W_resh)
                Conv_out[:,y,x,:]=c 
                
        Conv_out += self.B.value
        return Conv_out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

#         batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
                       
        X_pad = np.pad(self.X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        d_input_pad = np.zeros_like(X_pad)
        W_resh=self.W.value.reshape(-1, self.W.value.shape[-1])     

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                d_out_view = d_out[:,y,x,:]
                X_pad_view=X_pad[:,y:(y+self.filter_size),x:(x+self.filter_size),:]
                X_pad_view_resh = X_pad_view.reshape(X_pad_view.shape[0],-1)
                dw_xy = np.dot((X_pad_view_resh.T),                   d_out_view).reshape(self.filter_size,self.filter_size,self.in_channels,self.out_channels)
                self.W.grad += dw_xy
                
                d_input_xy = np.dot(d_out_view, W_resh.T)
                d_input_xy = d_input_xy.reshape(d_input_xy.shape[0], self.filter_size, self.filter_size, self.in_channels)
                d_input_pad[:,y:(y+self.filter_size),x:(x+self.filter_size),:] += d_input_xy 
                
        
        self.B.grad = np.sum(d_out.reshape(-1,d_out.shape[-1]), axis=0)
        d_input = d_input_pad[:,(0+self.padding):(d_input_pad.shape[1]-self.padding),(0+self.padding):(d_input_pad.shape[2]-self.padding),:] 
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = int(((height-self.pool_size)/self.stride)+1)
        out_width = int(((width-self.pool_size)/self.stride)+1)
        MaxPooling_out = np.zeros((batch_size, out_height, out_width, channels))
        self.ind = np.zeros(((out_height * out_width *batch_size), channels), dtype='int')
        i = 0
        for y in range(out_height): 
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                X_view=self.X[:,(y*self.stride):(y*self.stride+self.pool_size),(x*self.stride):(x*self.stride+self.pool_size),:] 
                X_view= X_view.reshape((X_view.shape[0],-1,X_view.shape[-1]))
                MaxPooling_out[:,y,x,:]=np.amax(X_view, axis=1)  
                ind_view = np.argmax(X_view, axis=1)
                self.ind[i*batch_size:(i+1)*batch_size,:] = ind_view   
                i += 1
        
        return MaxPooling_out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, __ = d_out.shape
        d_input = np.zeros_like(self.X)
        i = 0
        for y in range(out_height):
            for x in range(out_width):
                ind_view = self.ind[i*batch_size:(i+1)*batch_size,:]
                ind = np.array(np.unravel_index(ind_view, (self.pool_size,self.pool_size))).reshape(2,batch_size*channels)
                ind_0 = sorted((np.arange(batch_size).tolist())*channels) 
                ind_1 = ind[0]
                ind_2 = ind[1]
                ind_3 = (np.arange(channels).tolist())*batch_size
                d_input_view = np.zeros((batch_size, self.pool_size, self.pool_size, channels)) 
                d_input_view[ind_0, ind_1, ind_2, ind_3] = d_out[:,y,x,:].reshape(batch_size*channels)
                d_input[:,(y*self.stride):(y*self.stride+self.pool_size),(x*self.stride):(x*self.stride+self.pool_size),:] += d_input_view 
                i += 1
                        
        return d_input

      
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        Flattener_out = X.reshape((batch_size, height*width*channels))
        return Flattener_out

    def backward(self, d_out):
        # TODO: Implement backward pass
        d_input = d_out.reshape(self.X_shape)
        return d_input

    def params(self):
        # No params!
        return {}
