import numpy as np
import torch as t
import torchvision # to download MNIST data from pytorch datasets
import torchvision.transforms as transforms # to transform MNIST data into tensors
from torch.nn import functional as F # for one-hot encoding
from torch.utils.data import DataLoader # to load the datasets efficient
import torch.utils.data as td # for data split
from copy import deepcopy as dc # to prevent bugs between training & validation sets
# constant spsilon of float 64
EPS = t.finfo(t.float64).eps
LAMBDA_REG = 0.05

def initialize_parameters(layer_dims):
    t.manual_seed(888)
    params_dict = {}
    for i, layer_dim in enumerate(layer_dims[1::]):
        W = t.normal(mean=0, std=np.sqrt(2/layer_dims[i]), size=(layer_dim, layer_dims[i])).to(t.double) # he_init_dist
        params_dict[i+1] = ((W), t.zeros(layer_dim,1))

    return params_dict


def linear_forward(A, W, b):
    # A are of shape (n-1, m) where n is num of features / neurons
    #   and m is num of samples in this batch
    # W shape: (n,n-1)
    # b shape: (n,1)

    Z = (t.mm(W, A) + b) 
    return (Z, {'A':A, 'W': W, 'b': b})


def softmax(Z):
    # Z shape: (layer_dim, n_samples)
    
    global EPS
    A = t.nn.Softmax(dim=0)(Z) + EPS
    return A,{'Z':Z}
    exps = t.exp(Z)
    # sum_exps = exps.sum(axis = 0)
    A = exps / (exps.sum(axis=1) + 0.000001)


def relu(Z):
    # Z shape: (layer_dim, n_samples)
    return np.maximum(0, Z), {'Z':Z}


def linear_activation_forward(A_prev, W, B, activation):
    '''
    A_prev.shape: (n,m)
    W.shape: (curr_layer, prev_layer)
    '''
    Z, cache = linear_forward(A_prev, W, B)
    A, bullshit_cache = relu(Z) if activation == "relu" else softmax(Z)
    cache["Activation"] = A
    cache['Z'] = Z

    return A, cache


def L_model_forward(X, parameters, use_batchnorm):
    # use_batchnorm = True # test
    X = X.to(t.double)

    n_layers = len(parameters)
    caches = []

    # propagate the first layer
    Ai, cache = linear_activation_forward(X, parameters[1][0], parameters[1][1], "relu")
    if use_batchnorm:
        Ai = apply_batchnorm(Ai)
        

    caches.append(cache)

    # propagate all linear layers
    for i in range(2, n_layers):
        W, B = parameters[i]


        Ai, cache = linear_activation_forward(Ai, W, B, 'relu')

        if use_batchnorm:
            Ai = apply_batchnorm(Ai)
        
        caches.append(cache)

    # propagate the softmax layer
    AL, cache = linear_activation_forward(Ai, parameters[n_layers][0], parameters[n_layers][1], 'softmax')

    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y, params): # TODO: TEST
    # we assume that AL shape is (n_class, n_samples)
    # output (cost) shape: (m, 1)
    n_samples = AL.shape[1]

    AL = np.log(AL)
    cost = -(AL*Y).sum(axis=0).mean().item()

    # calc wieghts penalty
    L2_penalty = 0

    for i,pars in params.items():
        w,b = pars
        L2_penalty +=  t.sum(w**2).item()
    
    return cost + (0.5 * LAMBDA_REG *L2_penalty)



def apply_batchnorm(A): # TODO: TEST
    # A shape: (layer_dim, m samples)
    global EPS


    mean = t.mean(A, dim=1, keepdim=True)
    var = t.var(A, dim=1, unbiased=False, keepdim=True)
    norm_A = (A-mean) / t.sqrt(var + EPS)
    return norm_A


def Linear_backward(dZ, cache):
    # dz shape: (n, m)
    n_samples = dZ.shape[1]
    A_prev = cache["A"] # (n-1, m)

    dW = t.mm(dZ, A_prev.T) / n_samples
    db = dZ.mean(axis=1)
    
    dA_prev = t.mm(cache['W'].T, dZ) # shape: (n-1, m)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    # dA shape: (layer_dim, 1)
    
    # first compute dz
    if activation == 'relu':
        dZ = relu_backward(dA, cache)
    elif activation == 'softmax':
        # dZ = softmax_backward(dA, cache)
        # we dont use softmax backward because we already got dZ from L_model_backward
        dZ = dA
    else:
        raise Exception("wrong activation func:", activation)

    # then apply linear_backward()
    dA_prev, dW, db = Linear_backward(dZ, cache)

    return  dA_prev, dW, db


def relu_backward(dA, activation_cache):
    # dA shape: (layer_dim, m)
    Z = activation_cache['Z'] # (layer_dim, m)

    d_relu = (Z > 0).float() 
    dZ = dA * d_relu

    return dZ # (layer_dim, m)


def softmax_backward(dA, activation_cache):
    # dA shape: (layer_dim, m)
    A = activation_cache['Activation']

    softmax_der = A * (1-A) # = softmax(Z)[0] * (1-softmax(Z)[0])
    dZ = dA * softmax_der
    return dZ

def L_model_backward(AL, Y, caches):
    # AL shape: (n, m)
    # Y shape: (n, m)

    grads = {}

    # calc dZ of the last layer for each sample
    dZ = AL-Y

    # get grads of the output (softmax) layer
    dA_prev, dW, db = linear_activation_backward(dZ, caches[-1], 'softmax')
    grads[f'dA{len(caches)}'] = dA_prev
    grads[f'dW{len(caches)}'] = dW
    grads[f'db{len(caches)}'] = db

    # get grads of all the linear layers
    for i in range(len(caches)-2, -1, -1):
        dA_prev, dW, db = linear_activation_backward(dA_prev, caches[i], 'relu')
        grads[f'dA{i+1}'] = dA_prev
        grads[f'dW{i+1}'] = dW
        grads[f'db{i+1}'] = db

    return grads


def Update_parameters(parameters, grads, learning_rate):
    for i, params in parameters.items():
        dW = grads[f"dW{i}"] # shape (n, n-1)
        db = grads[f"db{i}"] # shape (n, m)
        dW = dW + LAMBDA_REG*params[0]
        
        W = params[0] - (learning_rate * dW)# TODO: check if minus or plus learning rate
        b = (params[1].T - (learning_rate * db)).T# TODO: check if minus or plus learning rate

        parameters[i] = (W, b)
    return parameters


def L_layer_model(X,Y, layers_dims, learning_rate, num_iterations, batch_size):
    batches_trained_count = 0
    costs = []
    val_losses = []
    n_samples = X.shape[1]
    batch_size = np.minimum(n_samples, batch_size)
    batch_idx = 0
    prev_val_cost = -1 # the validation cost in the previous iteration, we use this for the stop criterion

    # init parameters
    params = initialize_parameters(layers_dims)

    # split the data to training and validation sets
    val_size = int(0.2 * n_samples)  # 20% for validation
    
    train_size = n_samples - val_size

    train_indices, val_indices = td.random_split(range(n_samples), [train_size, val_size])
    x_validation_set = X[:, val_indices]
    y_validation_set = Y[:, val_indices]

    X = X[:, train_indices]
    Y = Y[:, train_indices]

    # iterate over the data with batches
    for epoch in range(num_iterations):
        for i in range(batch_size, X.shape[1], batch_size):
            # get X and Y of the current batch
            Xi = X[:, i-batch_size:i]
            Yi = Y[:, i-batch_size:i]

            # Forward propagation
            AL, caches = L_model_forward(Xi, params, use_batchnorm=True)

            # calc training cost
            cost = compute_cost(AL, Yi, params)

            # evaluate validation set (cost & acc)
            val_acc = predict(x_validation_set, y_validation_set, dc(params))
            Y_val_pred, _ = L_model_forward(x_validation_set, dc(params), use_batchnorm= True) # the error maybe becuase of batchnorm
            val_cost = compute_cost(Y_val_pred, y_validation_set, params)

            # Back propagation
            grads = L_model_backward(AL, Yi, caches)
            
            params = Update_parameters(params, grads, learning_rate)

            # save trainand val costs and may stop each 100 batch
            if batches_trained_count >= 100:
                costs.append(cost)
                val_losses.append(val_cost)
                batches_trained_count = 0

                # stopping critereon
                if abs(val_cost - prev_val_cost) < EPS:
                    print("\n\nValidation is not changing break\n\n")
                    break
                prev_val_cost = val_cost




            batches_trained_count += 1


            caches = []

        # log
        print(f"Epoch {epoch+1} | ->  Batch {batch_idx+1} val Accuracy: {val_acc}, val cost:{val_cost} train cost: {cost}")

    AT, caches = L_model_forward(X, params, use_batchnorm=True)
    
    # calc training cost (on all the data)
    cost = compute_cost(AT, Y, params)
    
    # log
    print(f"Training Acc: {predict(X, Y , params)}, all train cost: {cost}")

    return params, costs

def predict(X,Y, parameters):
    Y_pred, caches = L_model_forward(X, parameters, use_batchnorm= True) # the error maybe becuase of batchnorm

    Y_pred = np.argmax(Y_pred.tolist(), axis = 0)
    Y = np.argmax(Y.tolist(), axis = 0)

    accuracy = (Y == Y_pred).sum() / X.shape[1]

    return accuracy

