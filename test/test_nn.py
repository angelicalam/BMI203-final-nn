# TODO: import dependencies and write unit tests below
import numpy as np
from nn.nn import NeuralNetwork
from nn.io import read_text_file, read_fasta_file
from nn.preprocess import sample_seqs, one_hot_encode_seqs
from sklearn.metrics import log_loss
import warnings

def test_single_forward():
    nn_arch = [{'input_dim': 4, 'output_dim': 2, 'activation': 'relu'}, 
               {'input_dim': 2, 'output_dim': 4, 'activation': 'sigmoid'}]
    lr = 1
    seed = 42
    batch_size = 3
    epochs = 10
    loss_function = 'mse'
    # Generate NeuralNetwork. 
    model = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)
    
    W_curr = np.array([[ 1, 1, 1, 1],
                       [ 0, 0, 0, 0]])
    b_curr = np.array([[-10],
                       [ 1]])
    # shape (input_dim, batch_size)
    A_prev = np.ones((4, 3))
    A_curr, Z_curr = model._single_forward(W_curr, b_curr, A_prev, 'relu')
    
    assert A_curr.shape == (2,3) and A_curr.shape == Z_curr.shape, "A_curr, Z_curr shape is incorrect"
    assert np.array_equal(Z_curr, np.array([[4-10, 4-10, 4-10],
                                            [0+1, 0+1, 0+1]])), "Z_curr is incorrect"
    assert np.array_equal(A_curr, np.array([[0, 0, 0],
                                            [0+1, 0+1, 0+1]])), "A_curr is incorrect"


def test_forward():
    nn_arch = [{'input_dim': 4, 'output_dim': 2, 'activation': 'relu'}, 
               {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 1
    seed = 42
    batch_size = 3
    epochs = 10
    loss_function = 'mse'
    # Generate NeuralNetwork. 
    model = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)
    
    model._param_dict['W1'] = np.array([[ 1, 1, 1, 1],
                                        [ 0, 0, 0, 0]])
    model._param_dict['b1'] = np.array([[-10],
                                        [ 1]])
    model._param_dict['W2'] = np.array([[ 1, 1]])
    model._param_dict['b2'] = np.array([[ 0]])
    # shape (batch_size, input_dim=features)
    X = np.ones((3, 4))
    output, cache = model.forward(X)
    
    assert output.shape == (3,), "output shape is incorrect"
    # cache should have 2 Z matrices and 3 A matrices (A1 is the input)
    assert np.array_equal(list(cache.keys()), ['A1', 'A2', 'Z2', 'A3', 'Z3']), "cache is incorrect"
    assert np.array_equal(cache['A1'], X.T)
    # A2 and Z2 are identical to what was checked in test_single_forward
    # Check A3 and Z3 matrices
    assert np.array_equal(cache['Z3'], np.array([[1, 1, 1]])), "Z is incorrect"
    # A3 = sigmoid(Z3)
    assert np.array_equal(cache['A3'], 1/(1+np.exp(-1*np.array([[1, 1, 1]]))) ), "A is incorrect"
    assert np.array_equal(output, cache['A3'].flatten()), "output is incorrect"
    

def test_single_backprop():
    nn_arch = [{'input_dim': 4, 'output_dim': 2, 'activation': 'relu'}, 
               {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 1
    seed = 42
    batch_size = 3
    epochs = 10
    loss_function = 'mse'
    # Generate NeuralNetwork. 
    model = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)
    
    W_curr = np.array([[ 1, 1, 1, 1],
                       [ 0, 0, 0, 0]])
    b_curr = np.array([[-10],
                       [ 1]])
    # shape (input_dim, batch_size)
    A_prev = np.ones((4, 3))
    Z_curr = A_prev
    # shape (output_dim, batch_size)
    dA_curr = np.ones((2, 3))
    dA_prev, dW_curr, db_curr = model._single_backprop(W_curr, b_curr, Z_curr, 
                                                       A_prev, dA_curr, 'relu')
    # dA_prev ~ delta^(l) and dA_curr ~ delta^(l+1)
    # shape (input_dim, batch_size)
    assert dA_prev.shape == (4, 3), "dA_prev shape is incorrect"
    assert dW_curr.shape == W_curr.shape, "dW_curr shape is incorrect"
    assert db_curr.shape == b_curr.shape, "db_curr shape is incorrect"
    assert np.array_equal(dA_prev, np.ones((4, 3))), "dA_prev is incorrect"
    assert np.array_equal(dW_curr, np.ones((2, 4))*3), "dW_curr is incorrect"
    assert np.array_equal(db_curr, np.ones((2, 1))), "db_curr is incorrect"

    
def test_predict():
    nn_arch = [{'input_dim': 4, 'output_dim': 2, 'activation': 'relu'}, 
               {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 1
    seed = 42
    batch_size = 3
    epochs = 10
    loss_function = 'mse'
    # Generate NeuralNetwork. 
    model = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)
    
    model._param_dict['W1'] = np.array([[ 1, 1, 1, 1],
                                        [ 0, 0, 0, 0]])
    model._param_dict['b1'] = np.array([[-10],
                                        [ 1]])
    model._param_dict['W2'] = np.array([[ 1, 1]])
    model._param_dict['b2'] = np.array([[ 0]])
    # shape (batch_size, input_dim=features)
    X = np.ones((3, 4))
    output, cache = model.forward(X)
    # Before backpropagation, predict() should return the same output
    y_hat = model.predict(X)
    assert np.array_equal(output, y_hat), "predict returns to wrong output"
    

def test_binary_cross_entropy():
    nn_arch = [{'input_dim': 4, 'output_dim': 2, 'activation': 'relu'}, 
               {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 1
    seed = 42
    batch_size = 3
    epochs = 10
    loss_function = 'mse'
    # Generate NeuralNetwork. 
    model = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)
    X = np.ones((3, 4))
    y_hat = model.predict(X)
    y_true = np.array([1,1,0])
    loss = model._binary_cross_entropy(y_true, y_hat)
    # Use a different method to calculate the binary cross entropy
    sk_loss = log_loss(y_true, y_hat)
    assert np.allclose(loss, sk_loss), "BCE is incorrect"
    

def test_binary_cross_entropy_backprop():
    nn_arch = [{'input_dim': 4, 'output_dim': 2, 'activation': 'relu'}, 
               {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 1
    seed = 42
    batch_size = 3
    epochs = 10
    loss_function = 'mse'
    # Generate NeuralNetwork. 
    model = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)
    X = np.ones((3, 4))
    y_hat = model.predict(X)
    y_true = np.array([1,1,0])
    dloss = model._binary_cross_entropy_backprop(y_true, y_hat)
    # d_BCE = -y/A + (1-y)/(1-A)
    assert np.allclose(dloss, (-1*y_true/y_hat) + ((1-y_true)/(1-y_hat)) ), "BCE backprop is incorrect"
    

def test_mean_squared_error():
    nn_arch = [{'input_dim': 4, 'output_dim': 2, 'activation': 'relu'}, 
               {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 1
    seed = 42
    batch_size = 3
    epochs = 10
    loss_function = 'mse'
    # Generate NeuralNetwork. 
    model = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)
    loss = model._mean_squared_error(np.ones(3), np.zeros(3))
    assert loss == 1, "MSE is incorrect"
    

def test_mean_squared_error_backprop():
    nn_arch = [{'input_dim': 4, 'output_dim': 2, 'activation': 'relu'}, 
               {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 1
    seed = 42
    batch_size = 3
    epochs = 10
    loss_function = 'mse'
    # Generate NeuralNetwork. 
    model = NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)
    dloss = model._mean_squared_error_backprop(np.ones(3), np.zeros(3))
    assert np.array_equal(dloss, -2 * np.ones(3)), "MSE backprop is incorrect"


def test_sample_seqs():
    seqs = np.array(['AGTCG']*500 + ['TTTTT']*500)
    labels = np.array([1]*500 + [0]*500)
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)
    # Balanced classes are returned
    assert np.mean(sampled_labels) == 0.5, "Imbalanced classes"
    

def test_one_hot_encode_seqs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.array_equal(one_hot_encode_seqs(np.array(['AA'])), 
                              np.array([[1, 0, 0, 0, 1, 0, 0, 0]]))
    