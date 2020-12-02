import numpy as np


# 1. Linear_layer
class linear_layer:
    def __init__(self, input_D, output_D):
        self.params = dict()
        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))

        self.gradient = dict()
        # Initialize gradients with zeros
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros((1, output_D))

    def forward(self, X):
        forward_output = np.add(np.dot(X, self.params['W']), self.params['b'])
        return forward_output

    def backward(self, X, grad):
        # Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'], self.params['b'].
        self.gradient['W'] = np.dot(X.T, grad)
        self.gradient['b'] = np.sum(grad, axis=0)
        backward_output = np.dot(grad, self.params['W'].T)
        return backward_output


# 2. ReLU Activation
class relu:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        # relu forward pass. Store the result in forward_output    #
        forward_output = np.maximum(X, 0, X)
        return forward_output

    def backward(self, X, grad):
        backward_output = np.multiply(grad, (X > 0).view('i1'))
        return backward_output


# 3. tanh Activation
class tanh:

    def forward(self, X):
        forward_output = np.tanh(X)
        return forward_output

    def backward(self, X, grad):
        # Derivative of tanh is (1 - tanh^2)
        derivative_tanh = 1 - np.power(np.tanh(X), 2)
        backward_output = grad * derivative_tanh
        with open("backward", 'w') as f:
            f.write(backward_output)
        return backward_output


# 4. Dropout
class dropout:
    def __init__(self, r):
        self.r = r
        self.mask = None

    def forward(self, X, is_train):
        # If p >= self.r, output that element multiplied by (1.0 / (1 - self.r)); otherwise, output 0 for that element
        if is_train:
            self.mask = (np.random.uniform(0.0, 1.0, X.shape) >= self.r).astype(float) * (1.0 / (1.0 - self.r))
        else:
            self.mask = np.ones(X.shape)
        forward_output = np.multiply(X, self.mask)
        return forward_output

    def backward(self, X, grad):
        backward_output = np.multiply(grad, self.mask)
        return backward_output


# 5. Mini-batch Gradient Descent Optimization
def miniBatchGradientDescent(model, momentum, _lambda, _momentum_hyparameter, _learning_rate):
    # model: Dictionary containing all parameters of the model
    # momentum: Check add_momentum() function in utils.py to understand this parameter
    # _lambda: Regularization constant
    # _alpha: Momentum hyperparameter
    # _learning_rate: Learning rate for the update
    for module_name, module in model.items():

        # check if a module has learnable parameters
        if hasattr(module, 'params'):
            # print(module_name)
            # print(module.params)
            for key, _ in module.params.items():
                g = module.gradient[key] + _lambda * module.params[key]

                if _momentum_hyparameter > 0.0:
                    # m = alpha * m - learning_rate * g (Check add_momentum() function in utils file)
                    # And update model parameter
                    momentum_name = module_name + '_' + key
                    momentum[momentum_name] = _momentum_hyparameter * momentum[momentum_name] \
                                              - _learning_rate * g
                    module.params[key] += momentum[momentum_name]
                    # print(key)
                    # print(module.params[key])

                else:
                    module.params[key] -= _learning_rate * g

    return model
