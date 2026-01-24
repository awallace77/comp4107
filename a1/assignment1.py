# Name this file assignment1.py when you submit
# import torch
import math

# A function simulating an artificial neuron
def artificial_neuron(x, w):
  # x is a list of inputs of length n
  # w is a list of inputs of length n

  # Take weighted sum
  weighted_sum = sum([xi * wi for xi, wi in zip(x, w)])
  output = siLu(weighted_sum) 

  # output is the output from the neuron
  return output

def siLu(x):
  return x / (1 + math.exp(-x))


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


if __name__ == "__main__":

  # Question 1
  siLu_input = 2
  siLu_output = siLu(siLu_input)
  x = [1, 2, 3, 4]
  w = [1, 1, 1, 1]
  output = artificial_neuron(x, w)
  print(f"QUESTION 1: siLu({siLu_input})")
  print(f"Expected: 1.76159")
  print(f"Actual  : {siLu_output}" )
  print("")
  print(f"QUESTION 1: artificial_neuron")
  print(f"Expected: ?")
  print(f"Actual  : {output}")