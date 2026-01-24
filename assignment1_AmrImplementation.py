# Name this file assignment1.py when you submit
#import torch
import math

# A function simulating an artificial neuron
def artificial_neuron(x, w):
  # Aggregation Function:SW
  inp = 0
  for i in range(len(x)):
    inp += x[i]*w[i]
  
  # Activation Function
  output = inp / (1 + math.exp(-inp))
  
  return output


# A function performing gradient descent
def gradient_descent(f, df, x0, alpha):
  # f is a function that takes as input a list of length n
  # df is the gradient of f; it is a function that takes as input a list of length n
  # x0 is an initial guess for the input minimizing f
  # alpha is the learning rate

  # argmin_f is the input minimizing f
  # min_f is the value of f at its minimum
  return argmin_f, min_f


# A function that returns a neural network module in PyTorch
def pytorch_module():

  # A pytorch module
  return module

x = [1,1,1]
w = [2,1,4]
print(artificial_neuron(x,w))