# Name this file assignment1.py when you submit
import numpy as np
import math
import torch

# Helper functions
def weighted_sum(x, w):
  return sum([xi * wi for xi, wi in zip(x, w)])

def siLu(x):
  return x / (1 + math.exp(-x))

def d_siLu(x):
  exp_neg = math.exp(-x)
  return (1 / (1 + exp_neg)) + (x * exp_neg) / ((1 + exp_neg)**2)


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
  maxIterations = 10000
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


def q4a_gradient_descent():
  """ Experiments for Q4a """

  x_1 = 3
  x_2 = -2
  y_1 = 0.5
  y_2 = -0.75

  alphas = [0.1, 0.05, 0.01, 0.005, 0.001]
  x0 = [0, 0]

  def f(input):
    a = input[0]
    b = input[1]
    return 1/2 * ((a * x_1 + b - y_1)**2 + (a * x_2 + b - y_2)**2)
  
  def df(input):
    a = input[0]
    b = input[1]
    df_da = x_1 * (a * x_1 + b - y_1) + x_2 * (a * x_2 + b - y_2)
    df_db = (a * x_1 + b - y_1) + (a * x_2 + b - y_2)
    return [df_da, df_db]

  for alpha in alphas:
    print(f"[TESTING]: Q4A - gradient_descent (lr = {alpha})")
    print(f"Actual  : {gradient_descent(f, df, x0, alpha)}" )
    print("")



def q4b_gradient_descent():
  """ Experiments for Q4b """

  x_1 = 3
  x_2 = -2
  y_1 = 0.5
  y_2 = -0.75

  alphas = [0.1, 0.05, 0.01, 0.005, 0.001]
  x0 = [0, 0]

  def f(input):
    a = input[0]
    b = input[1]
    return 1/2 * ((siLu(a*x_1 + b) - y_1)**2 + (siLu(a*x_2 + b) - y_2)**2)
  
  def df(input):
    a = input[0]
    b = input[1]

    z_1 = weighted_sum([x_1, 1], [a, b])
    z_2 = weighted_sum([x_2, 1], [a, b])

    e_1 = siLu(z_1) - y_1
    e_2 = siLu(z_2) - y_2

    df_da = (e_1 * d_siLu(z_1) * x_1 + e_2 * d_siLu(z_2) * x_2)

    df_db = (e_1 * d_siLu(z_1) +e_2 * d_siLu(z_2))

    return [df_da, df_db]


  for alpha in alphas:
    print(f"[TESTING]: Q4B - gradient_descent (lr = {alpha})")
    print(f"Actual  : {gradient_descent(f, df, x0, alpha)}" )
    print("")


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

  # Question 4
  q4a_gradient_descent()
  q4b_gradient_descent()