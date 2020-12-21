"""
A machine learning package with one hidden layer.
Version 21/12/2020
"""
# Package imports
import json

import numpy as np
import matplotlib.pyplot as plt


# np.random.seed(1) # set a seed so that the results are consistent
def layer_sizes(X, Y, h=4):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (≈ 3 lines of code)
    n_x = X.shape[1]  # size of input layer
    n_h = h
    n_y = Y.shape[1]  # size of output layer
    ### END CODE HERE ###
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    ### END CODE HERE ###

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


sig = lambda x: 1 / (1 + np.exp(-x))


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###

    # Implement Forward Propagation to calculate A2 (probabilities)
    ### START CODE HERE ### (≈ 4 lines of code)
    Z1 = W1 @ X.T
    A1 = np.tanh(Z1 + b1)
    Z2 = W2 @ A1
    A2 = sig(Z2 + b2)
    ### END CODE HERE ###
    #     print(A2.shape)
    #     assert(A2.shape == (Y.shape[0], X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cost(cross- entropy cost) given in the equation above.

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost --cross-entropy cost
    """
    A2 = A2.T
    m = Y.shape[0]  # number of example
    # Compute cost
    ### START CODE HERE ### (≈ 2 lines of code)
    cost = -(1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    #     print(A2)
    ### END CODE HERE ###
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17

    assert (isinstance(cost, float))
    return cost


def TanHTag(z):
    return 1 - np.tanh(z) ** 2


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[0]
    #     print(X.shape)

    # First, retrieve W1 and W2 from the dictionary "parameters".
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters['W1']
    W2 = parameters['W2']
    ### END CODE HERE ###

    # Retrieve also A1 and A2 from dictionary "cache".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1 = cache['A1']
    A2 = cache['A2']
    ### END CODE HERE ###
    #     A2 = A2.T
    # Backward propagation: calculate dW1, db1, dW2, db2.
    ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dZ2 = A2 - Y.T
    #     print(dZ2.shape, A1.shape)
    dW2 = 1 / m * dZ2 @ A1.T
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T @ dZ2 * TanHTag(cache['Z1'])
    dW1 = 1 / m * dZ1 @ X
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    ### END CODE HERE ###

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=0.5):
    """
    Updates parameters using the gradient accent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    ### END CODE HERE ###

    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    ## END CODE HERE ###

    # Update rule for each parameter
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


from sklearn.model_selection import *


def split():
    pass


split = train_test_split


def fix_targets(Y, Number_of_types):
    """
    Arguments:
    Y -- labels array
    Number_of_types -- number of types of targets
    The function returns target data of shape [0,0,0,1....]. use before splitting.
    """
    numbers_targets_og = []
    for i in Y:
        n = [0 for j in range(Number_of_types)]
        n[int(i)] = 1
        numbers_targets_og.append(n)
    return np.array(numbers_targets_og)


def normalize(X):
    """
    Arguments:
    X -- dataset array
    The function return a normalized dataset array. use before splitting.
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)


def nn_model(X, Y, n_h, num_iterations=None, print_cost=False, alpha=0.1, graph=False, Max_last_best=2500):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    alpha -- alpha variable
    print_cost -- if True, print the cost every 1000 iterations
    graph -- if True, plots a graph of costs over the course of the learning
    Max_last_best -- the number of iterations to do before finding a new minimum
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    # print(X.shape, Y.shape)

    X, X_validation, Y, Y_validation = split(X, Y, test_size=0.33)

    # print(X.shape, Y.shape, X_validation.shape, Y_validation.shape)

    # np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    ### START CODE HERE ### (≈ 5 lines of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###

    error_history = []
    error_validation_history = []
    best_error_validation = None
    best_NN = None
    last_best = 0

    # Loop (gradient descent)
    i = 0
    while 420:

        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, alpha)

        ### END CODE HERE ###
        # this_error = np.sum((sgn(A2).T - Y)**2)
        error_history.append(cost)

        # Validator shit
        A2_validation, cache_validation = forward_propagation(X_validation, parameters)
        # this_error_validation = np.sum((sgn(A2_validation).T - Y_validation)**2)
        cost_validation = compute_cost(A2_validation, Y_validation, parameters)
        error_validation_history.append(cost_validation)

        last_best += 1
        if best_NN == None or best_error_validation > cost_validation:
            best_NN = parameters
            best_error_validation = cost_validation
            last_best = 0

        # Print the cost every 1000 iterations
        if print_cost and i % 5 == 0:
            print("Cost after iteration %s:\tTrainer %f,\tValitation %f" % (
            i if i != 0 else "0000", cost, cost_validation))

        if num_iterations and i > num_iterations:
            break
        if last_best > Max_last_best:
            break
        i += 1

    print(f"Min trainer cost - {min(error_history)}, at {error_history.index(min(error_history))}")
    print(f"Min validation cost - {best_error_validation}, at {error_validation_history.index(best_error_validation)}")

    if graph:
        plt.plot(error_history, 'b-')
        plt.plot(error_validation_history, 'g-')
        print("Green - Validation")
        print("Blue - Train")

    return best_NN


def sgn(x):
    return np.array(x >= 0.5, dtype=int)


def maxIndex(X):
    return np.array([i == np.max(i) for i in X], dtype=int)


def model_test(parameters, X, Y, do_return=False, ChooseMax=True):
    """
    Arguments:
    parameters -- array of weights to test the NN with
    X -- array of test dataset
    Y -- array of test labels
    do_return(optional,default is False) -- wether or not to return the success rate
    The function tests the NN's preformence for a given weights array
    """
    results, lo = forward_propagation(X, parameters)
    if ChooseMax:
        error = maxIndex(results.T) - Y
    else:
        error = sgn(results).T - Y
    success = sum(not (sum(abs(j) for j in i)) for i in error)
    print(f"success rate: {round(100 * success / len(Y), 2)}%")
    print(f"successfuly guessed {success} / {len(Y)} examples correctly")
    if do_return:
        return 100 * success / len(Y)


def apply_on_example(arr, parameters):
    """
    Arguments:
    arr -- an array with data of one example
    parameters -- array of weights to test the NN on
    The function returns its guess given a weights array
    """
    results, lo = forward_propagation(arr, parameters)
    l = list(results)
    print(f"the nn's guess: {l.index(max(results))}")
    print('\n\nhere are the guesses the nn was most confident about(marked with <----):')
    for i in range(len(l)):
        print(f"{i} {'<----' if sgn(results)[i] else ''}")


def SaveDict(parameters, name="NN"):
    dictlist = {}
    for key in parameters:
        dictlist[key] = [[j for j in i] for i in parameters[key]]

    with open(name+'.json', 'w') as fp:
        json.dump(dictlist, fp)


def ReadDict(name="NN"):
    with open(name+'.json', 'r') as fp:
        dictlist = json.load(fp)

    parameters = {}
    for key in dictlist:
        parameters[key] = np.array(dictlist[key])

    return parameters