# Name this file assignment1.py when you submit
#import torch
import math

# A function simulating an artificial neuron
def artificial_neuron(x, w):
  # Aggregation Function:
  inp = 0
  for i in range(len(x)):
    inp += x[i]*w[i]
  
  # Activation Function
  output = inp / (1 + math.exp(-inp))
  
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
    if Math.sqrt(gradientNorm) <= tolerance:
      break

    for j in range(len(argmin_f)):
      argmin_f[j] = argmin_f[j] - alpha*gradient[j]

  # argmin_f is the input minimizing f
  # min_f is the value of f at its minimum
  min_f = f(argmin_f)
  return argmin_f, min_f


# A function that returns a neural network module in PyTorch
def pytorch_module():

  # A pytorch module
  return module

x = [1,1,1]
w = [2,1,4]
print(artificial_neuron(x,w))