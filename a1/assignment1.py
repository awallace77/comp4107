# Name this file assignment1.py when you submit
import torch
import numpy as np
import math

# Helper functions
def weighted_sum(x, w):
  return sum([xi * wi for xi, wi in zip(x, w)])

def siLu(x):
  return x / (1 + math.exp(-x))


# A function simulating an artificial neuron
def artificial_neuron(x, w):
  # x is a list of inputs of length n
  # w is a list of inputs of length n

  if (x is None or w is None) or (len(x) != len(w)):
    return

  # Take weighted sum
  w_sum = weighted_sum(x, w)
  output = siLu(w_sum)

  # output is the output from the neuron
  return output

# A function performing gradient descent
def gradient_descent(f, df, x0, alpha):
  tolerance = 0.0001
  maxIterations = 1000
  # f is a function that takes as input a list of length n
  # df is the gradient of f; it is a function that takes as input a list of length n
  # x0 is an initial guess for the input minimizing f
  # alpha is the learning rate
  argmin_f = list(x0)

  for i in range(maxIterations):
    gradient = df(argmin_f)
    gradientNorm = 0
    for j in gradient:
      gradientNorm += j**2
    if math.sqrt(gradientNorm) <= tolerance:
      break

    for j in range(len(argmin_f)):
      argmin_f[j] = argmin_f[j] - alpha*gradient[j]

  # argmin_f is the input minimizing f
  # min_f is the value of f at its minimum
  min_f = f(argmin_f)
  return argmin_f, min_f

# A function that returns a neural network module in PyTorch
def pytorch_module():

  class NeuralNetwork(torch.nn.Module):
    """ A simple Neural Network with 2 hidden layers and 1 output layer """
    def __init__(self):
      super().__init__()
      self.flatten = torch.nn.Flatten()
      self.linear_relu_stack = torch.nn.Sequential(
        torch.nn.Linear(28*28, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
      )
   
    def forward(self, x):
      x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return logits
     
  # A pytorch module
  module = NeuralNetwork()
  return module

## Testing functions ## 
def test_artificial_neuron(x, w, expected):
  print(f"[TESTING]: artificial_neuron")
  print(f"Expected: {expected}")
  print(f"Actual  : {artificial_neuron(x, w)}" )
  print("")

if __name__ == "__main__":
  
  # Question 1: TESTING siLu 
  siLu_input = 2
  siLu_output = siLu(siLu_input)
  # print(weighted_sum(x, w))
  print(f"QUESTION 1: siLu({siLu_input})")
  print(f"Expected: 1.76159")
  print(f"Actual  : {siLu_output}" )
  print("")

  # Question 1: INVALID INPUT
  x = None
  w = [1]
  test_artificial_neuron(x, w, None)
   
  x = [1, 2, 3, 4]
  w = [1, 1, 1, 1]
  test_artificial_neuron(x, w, 9.99954)

  x = [2, 1, 8, 3]
  w = [0.4, 0.5, 0.2, 0.9]
  # print(weighted_sum(x, w))
  test_artificial_neuron(x, w, 5.57936)

  # Question 3
  module = pytorch_module()
  print(module)