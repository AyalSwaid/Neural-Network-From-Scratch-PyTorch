# Neural-Network-From-Scratch-PyTorch
Building deep neural network model from scratch using pytorch (only tensors), then test it on MNIST dataset 

## How to run
pass layers diminsions to initialize_parameters function and get layers weights, then train the model using L_layer_model function. 

## Files:
- DNN.py - Standad Deep neural network model built from scratch, including forward and back proppagation, Batchnorm, loss, prediction, softmax.
- L2.py - Same exact model but with L2 regularization.

## Results on MNIST
Model used: L2 model, lambda=0.05, learning rate=0.09, layers_dims(aside from the input layer)=[20,7,5,10]
### Accuracy and loss
Train acc: 0.91 | Train Loss: 0.33
Test acc: 0.90 | Test Loss: 0.35
### Learning rate
![image](https://github.com/AyalSwaid/Neural-Network-From-Scratch-PyTorch/assets/57876635/85464634-f805-4730-8bc0-1470385474c1)
