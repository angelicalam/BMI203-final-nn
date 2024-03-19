# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    
    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # Use the correct activation function
        if activation == 'sigmoid':
            activation = self._sigmoid
        elif activation == 'relu':
            activation = self._relu
        else:
            raise ValueError(f"Not an activation function used by the model.")
            
        # Calculate current layer linear transform matrix.
        z = (W_curr @ A_prev) + b_curr 
        # Return both activation matrix and linear transform matrix
        return activation(z), z

    
    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        A_prev = X.T   # Input for first layer
        # Initialize cache. X serves as "A_prev" for backprop of W1, b1
        cache = {'A1': A_prev}
        
        # Iterate through layers of the specified neural architecture
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            activation = layer['activation']
            # Perform forward pass step
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)
            # Store Z and A matrices for use in backprop
            cache['A' + str(layer_idx+1)] = A_curr 
            cache['Z' + str(layer_idx+1)] = Z_curr
            A_prev = A_curr
        # After iterating through all layers, A_curr is the output of the full forward pass.
        # A_curr needs to be reshaped to (batch_size, output_dim)
        output = A_curr.T
        # If output_dim is 1, output should be shape (batch_size,)
        if self.arch[-1]['output_dim'] == 1:
            output = output.flatten()
            
        return output, cache


    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # Use the correct activation function derivative
        if activation_curr == 'sigmoid':
            d_activation = self._sigmoid_backprop
        elif activation_curr == 'relu':
            d_activation = self._relu_backprop
        else:
            raise ValueError(f"Not an activation function used by the model.")
        
        # Here, dA_prev ~ delta^(l) and dA_curr ~ delta^(l+1), but
        # W_curr, b_curr, Z_curr, and A_prev correspond to layer (l).
        # dA_prev = matrix Hadamard product of (W_curr^T @ dA_curr) and f'(Z_curr),
        # where f'(Z_curr) depends on the activation function used.
        dA_prev = (W_curr.T @ dA_curr) * d_activation(A_prev, Z_curr)
        # Compute partial derivatives of weights and bias using d_error_term^(l+1), i.e, dA_curr
        dW_curr = dA_curr @ A_prev.T
        db_curr = dA_curr.mean(axis=1, keepdims=True)   # Use mean gradient for mini-batch operations
        # Dimensions check: W_curr,dW_curr = (m, s); A_prev,dA_prev = (s, x); A_curr,dA_curr = (m, x)
        
        return dA_prev, dW_curr, db_curr

    
    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # Use the correct loss function derivative
        if self._loss_func == 'mse':
            d_loss = self._mean_squared_error_backprop
        elif self._loss_func == 'bce':
            d_loss = self._binary_cross_entropy_backprop
        else:
            raise ValueError(f"Not a loss function used by the model. Specify mse (mean squared error) or bce (binary cross entropy).")
          
        # Get output layer Z matrix, Z^(n_l). The A matrix here is the model output, y_hat
        Z_out = cache['Z' + str(len(self.arch)+1)]
        # Use the correct activation function derivative
        layer_out = self.arch[-1]
        activation_out = layer_out['activation']
        if activation_out == 'sigmoid':
            d_activation = self._sigmoid_backprop
        elif activation_out == 'relu':
            d_activation = self._relu_backprop
        else:
            raise ValueError(f"Not an activation function used by the model.")
            
        # Perform backprop on output layer.
        # Loss gradient needs to be reshaped to (output_dim, batch_size)
        grad_loss = d_loss(y, y_hat).reshape(layer_out['output_dim'], self._batch_size)
        dA_curr = grad_loss * d_activation(y_hat, Z_out)
        
        # Perform backprop on hidden layers
        grad_dict = dict()
        for idx in reversed(range(1, len(self.arch))):
            layer_idx = idx + 1
            # Get hidden layer activation
            activation = self.arch[idx-1]['activation']
            # Here, dA_prev ~ delta^(l) and dA_curr ~ delta^(l+1),
            # W_curr, b_curr, Z_curr, A_prev correspond to layer (l)
            Z_curr = cache['Z' + str(layer_idx)]
            A_prev = cache['A' + str(layer_idx)]
            W_l = 'W' + str(layer_idx)
            W_curr = self._param_dict[W_l]
            b_l = 'b' + str(layer_idx)
            b_curr = self._param_dict[b_l]
            # Calculate gradients
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, 
                                                              dA_curr, activation)
            grad_dict[W_l] = dW_curr 
            grad_dict[b_l] = db_curr
            # Back-pass delta
            dA_curr = dA_prev
            
        # Perform backprop for W1, b1
        W_curr = self._param_dict['W1']
        b_curr = self._param_dict['b1']
        A_prev = cache['A1']
        dW_curr = dA_curr @ A_prev.T
        db_curr = dA_curr.mean(axis=1, keepdims=True)
        grad_dict['W1'] = dW_curr 
        grad_dict['b1'] = db_curr
            
        return grad_dict


    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            # Update weight matrix for layer_idx
            W_l = 'W' + str(layer_idx)
            dW = grad_dict[W_l]
            self._param_dict[W_l] -= self._lr * dW/self._batch_size   # no regularization
            # Update bias vector for layer_idx
            b_l = 'b' + str(layer_idx)
            db = grad_dict[b_l]
            self._param_dict[b_l] -= self._lr * db/self._batch_size

    
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # Use the correct loss function
        if self._loss_func == 'mse':
            loss_f = self._mean_squared_error
        elif self._loss_func == 'bce':
            loss_f = self._binary_cross_entropy
        else:
            raise ValueError(f"Not a loss function used by the model. Specify mse (mean squared error) or bce (binary cross entropy).")
            
        per_epoch_loss_train = np.zeros(self._epochs)
        per_epoch_loss_val = np.zeros(self._epochs)
        
        for n in range(self._epochs):
            # Take a mini-batch from the training set
            batch_arg = np.random.choice(range(len(X_train)), self._batch_size, replace=False)
            X_batch = X_train[batch_arg,:]
            y_batch = y_train[batch_arg]
            # Run the training mini-batch through a forward pass
            y_hat, cache = self.forward(X_batch)
            # Then calculate the gradient of the loss via backprop
            grad_dict = self.backprop(y_batch, y_hat, cache)
            # Update parameters using calculated gradients
            self._update_params(grad_dict)
            # Calculate and store the training and validation loss for the current epoch
            per_epoch_loss_train[n] = loss_f(y_batch, y_hat)
            y_hat_val = self.predict(X_val)
            per_epoch_loss_val[n] = loss_f(y_val, y_hat_val)
            
        return per_epoch_loss_train, per_epoch_loss_val
    
    
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        return y_hat
        

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1 / (1 + np.exp(-1 * Z))
        return nl_transform
        

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = self._sigmoid(Z) * (1 - self._sigmoid(Z))
        return dZ
    

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.where(Z > 0, Z, 0)
        return nl_transform
    

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = np.where(Z > 0, 1, 0)
        return dZ
        

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # Binary cross entropy = −(𝑦log(𝑝)+(1−𝑦)log(1−𝑝))
        # This equation outputs a value that is high when the probability
        # is low, and decays exponentially as the probability approaches 1. 
        loss = np.mean( -1 * ((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat))) )
        return loss


    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # First, calculate the derivative of the loss for all weights
        # BCE = −(𝑦log( A matrix ) +(1−𝑦)log(1 - A matrix ))
        # d(BCE)/dA : -y/A + (1-y)/(1-A)
        dA = -1*y/y_hat + (1-y)/(1-y_hat)
        return dA
    

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        loss = np.mean((y - y_hat)**2)
        return loss
        

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = -2 * (y - y_hat)
        return dA
    