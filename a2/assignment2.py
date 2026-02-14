# Name this file assignment2.py when you submit
import numpy
import torch
import math
import matplotlib.pyplot as plt

# QUESTION 1 & 2: MULTI-TASK
class MultitaskNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # Code for constructor goes here

    # Activation
    self.relu = torch.nn.ReLU()
    self.softmax= torch.nn.Softmax(dim=1)

    # Normalization 
    self.normalization = torch.nn.LayerNorm(3)

    # Layers
    self.layer1 = torch.nn.Linear(3, 5)
    self.layer2 = torch.nn.Linear(5, 4)
    self.output1 = torch.nn.Linear(4, 3)
    self.output2 = torch.nn.Linear(4, 3)

  def forward(self, x):
    # Code for forward method goes here
    x = self.normalization(x)
    x = self.layer1(x)
    x = self.relu(x)
    x = self.layer2(x)
    x = self.relu(x)
    y_pred_a = self.softmax(self.output1(x))
    y_pred_b = self.softmax(self.output2(x))

    return y_pred_a, y_pred_b 


def multitask_training(data_filepath):
  """
    Trains the mutlitask network on training data
      - A loss function, which is computed as a sum of categorical cross-entropy losses for each of the three-class classification tasks.
      - An optimizer using stochastic gradient descent with a cosine learning rate schedule.
      - Each row in the file refers to an instance (i.e. an MLB position player).
  """
  num_epochs = 1000
  batch_size = 4

  data = numpy.loadtxt(data_filepath, delimiter=",")
  
  batches_per_epoch = math.ceil(data.shape[0] / batch_size)

  multitask_network = MultitaskNetwork()

  # Define loss function(s) here
  loss_func = torch.nn.CrossEntropyLoss()

  # Define optimizer here
  lr = 0.01
  optimizer = torch.optim.SGD(multitask_network.parameters(), lr=lr, momentum=0.9)
  # optimizer = torch.optim.Adam(multitask_network.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

  for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_index in range(batches_per_epoch):
      
      start = batch_index * batch_size
      end = start + batch_size
      
      batch = data[start:end]

      x = torch.as_tensor(batch[:, 6:9], dtype=torch.float32)
      y_a = torch.as_tensor(batch[:, 0:3], dtype=torch.float32)
      y_b = torch.as_tensor(batch[:, 3:6], dtype=torch.float32)

      y_pred_a, y_pred_b = multitask_network(x)

      # Compute loss 
      loss_a = -torch.sum(y_a * torch.log(y_pred_a + 1e-8)) / batch_size
      loss_b = -torch.sum(y_b * torch.log(y_pred_b + 1e-8)) / batch_size

      loss = loss_a + loss_b

      # Compute gradients 
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()
    
    # Update parameters according to SGD with learning rate schedule here
    scheduler.step()

  # A trained torch.nn.Module object
  return multitask_network

def multitask_evaluation(model, filepath):
  data = numpy.loadtxt(filepath, delimiter=",")
  x = torch.as_tensor(data[:, 6:9], dtype=torch.float32)
  y_a = torch.argmax(torch.as_tensor(data[:, 0:3]), dim=1)
  y_b = torch.argmax(torch.as_tensor(data[:, 3:6]), dim=1)
  
  with torch.no_grad():
    pred_a, pred_b = model(x)
    pred_a = torch.argmax(pred_a, dim=1)
    pred_b = torch.argmax(pred_b, dim=1)
    
    acc_a = (pred_a == y_a).float().mean()
    acc_b = (pred_b == y_b).float().mean()

    print("Task A Accuracy:", acc_a.item())
    print("Task B Accuracy:", acc_b.item())


# QUESTION 3: MLB Network
class MLBNetworkBase(torch.nn.Module):
  def __init__(self, input_dim, hidden_layers, activation_func):
    super().__init__()
    # Code for constructor goes here

    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_layers:
      layers.append(torch.nn.Linear(prev_dim, hidden_dim ))
      layers.append(activation_func)
      prev_dim = hidden_dim

    layers.append(torch.nn.Linear(prev_dim, 1))
    self.model = torch.nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x) 

class MLBNetwork(torch.nn.Module):
    def __init__(self):
      super().__init__()

      self.model = MLBNetworkBase(16, [16, 64, 32], torch.nn.ReLU()) 

    def forward(self, x):

      return self.model(x)

def mlb_position_player_salary(filepath):
  """
    A function that creates a pytorch model to predict the salary of an MLB position player
    
    :param filepath: is the path to an csv file containing the dataset
  """
  num_epochs = 200
  batch_size = 8

  data = numpy.loadtxt(filepath, delimiter=",", skiprows=1)
  
  # Shuffle data
  perm = numpy.random.permutation(data.shape[0]) 
  data = data[perm]

  # Define model
  model = MLBNetwork()

  # Loss & optimizer
  lr = 0.001
  loss_func = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model .parameters(), lr=lr)

  # Split training and test data
  Y = data[:, 0:1]
  X = data[:, 1:]

  # Training & validation split
  split = int(0.8 * len(X))
  X_train, X_val = X[:split], X[split:]
  Y_train, Y_val = Y[:split], Y[split:]

  # Remove outliers (top 1% of salaries)
  threshold = numpy.percentile(Y_train, 99)
  mask = Y_train.flatten() < threshold
  X_train, Y_train = X_train[mask], Y_train[mask]

  # Normalize inputs
  mean = X_train.mean(axis=0) 
  std = X_train.std(axis=0) + 1e-8
  X_train = (X_train - mean) / std
  X_val   = (X_val - mean) / std

  # Convert to tensors
  X_train = torch.tensor(X_train, dtype=torch.float32)
  Y_train = torch.tensor(Y_train, dtype=torch.float32)
  X_val   = torch.tensor(X_val, dtype=torch.float32)
  Y_val   = torch.tensor(Y_val, dtype=torch.float32)

  # Train model
  batches_per_epoch = math.ceil(len(X_train) / batch_size)
  model.train()
  for epoch in range(num_epochs):
    
    epoch_loss = 0.0

    for batch_index in range(batches_per_epoch):
      
      # Get batch 
      start = batch_index * batch_size
      end = start + batch_size
      x = torch.as_tensor(X_train[start:end, :], dtype=torch.float32)
      y = torch.as_tensor(Y_train[start:end, :], dtype=torch.float32)

      # Prediction & loss
      y_pred = model(x)
      loss = loss_func(y_pred, y)

      # Backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      epoch_loss += loss.item()

    # if epoch % 20 == 0:
    #   print(f"Epoch: {epoch} || Loss: {epoch_loss / (batch_size * batches_per_epoch)}")

  # Evaluation
  model.eval()
  with torch.no_grad():
    Y_val_pred = model(X_val)
    loss = loss_func(Y_val_pred, Y_val)

    # RMSE
    validation_performance = torch.sqrt(loss)

  return model, validation_performance.item()

def mlb_experiments(filepath):

  # Load data  
  data = numpy.loadtxt(filepath, delimiter=",", skiprows=1)
  Y = data[:, 0:1]
  X = data[:, 1:]

  # Split data: 70% train, 15% val, 15% test
  n = len(X)
  train_end = int(0.7 * n)
  val_end = int(0.85 * n)
  X_train, Y_train = X[:train_end], Y[:train_end]
  X_val,   Y_val   = X[train_end:val_end], Y[train_end:val_end]
  X_test,  Y_test  = X[val_end:], Y[val_end:]

  # Normalize
  mean = X_train.mean(axis=0)
  std = X_train.std(axis=0) + 1e-8
  X_train = (X_train - mean) / std
  X_val   = (X_val - mean) / std
  X_test  = (X_test - mean) / std
  
  # Convert to tensors
  X_train = torch.tensor(X_train, dtype=torch.float32)
  Y_train = torch.tensor(Y_train, dtype=torch.float32)
  X_val   = torch.tensor(X_val, dtype=torch.float32)
  Y_val   = torch.tensor(Y_val, dtype=torch.float32)
  X_test  = torch.tensor(X_test, dtype=torch.float32)
  Y_test  = torch.tensor(Y_test, dtype=torch.float32)


  def train(hidden_layers, activation_func, num_epochs, lr=0.001):
  
    model = MLBNetworkBase(16, hidden_layers, activation_func)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
      model.train()
      optimizer.zero_grad()

      Y_pred = model(X_train)
      loss = loss_func(Y_pred, Y_train)
      loss.backward()
      optimizer.step()

      model.eval()
      with torch.no_grad():
        val_pred = model(X_val)
        val_loss = loss_func(val_pred, Y_val)

      train_losses.append(torch.sqrt(loss).item())
      val_losses.append(torch.sqrt(val_loss).item())

    return model, train_losses, val_losses
  
  model_performances = {
    "neurons": {
      "name": "",
      "loss": math.inf
    },
    "layers": {
      "name": "",
      "loss": math.inf
    },
    "epochs": {
      "name": "",
      "loss": math.inf
    },
    "activation": {
      "name": "",
      "loss": math.inf
    }
  }
  
  # a) Number of neurons in each hidden layer
  neurons = [8, 16, 32, 64, 128]
  val_results = []

  for n in neurons:
    model, _, val_losses = train([n, n], torch.nn.ReLU(), 200)
    val_results.append(val_losses[-1])
    if val_losses[-1] < model_performances["neurons"]["loss"]:
      model_performances["neurons"]["name"] = n
      model_performances["neurons"]["loss"] = val_losses[-1]

  plt.figure(figsize=(8, 5)) 
  plt.plot(neurons, val_results, marker='o', linestyle='-', color='blue')
  plt.xlabel("Width of Hidden Layer")
  plt.ylabel("Validation Performance (RMSE)")
  plt.title("Validation Performance of Different Number of Neurons")
  plt.show()
  # plt.savefig("neurons.png")
  # plt.clf() 

  # b) Number of hidden layers in network
  width = 32
  num_layers = 4
  layers = []

  for i in range(num_layers):
    layers.append([width for j in range(i)])

  val_results = []
  
  for i in range(num_layers):
    model, _, val_losses = train(layers[i], torch.nn.ReLU(), 200)
    val_results.append(val_losses[-1])
    if val_losses[-1] < model_performances["layers"]["loss"]:
      model_performances["layers"]["name"] = layers[i] 
      model_performances["layers"]["loss"] = val_losses[-1]

  plt.figure(figsize=(8, 5)) 
  plt.plot([k + 1 for k in range(num_layers)], val_results, marker='o', linestyle='-', color='blue')
  plt.xlabel(f"Number of Hidden Layers (width = {width})")
  plt.ylabel("Validation Performance (RMSE)")
  plt.title("Validation Performance of Different Number Layers")
  plt.show()
  # plt.savefig("layers.png")
  # plt.clf() 

  # c) Number of epochs
  num_epochs = [100, 200, 400, 800]
  val_results = []
  for epochs in num_epochs:
    model, _, val_losses = train([64, 32], torch.nn.ReLU(), epochs)
    val_results.append(val_losses[-1])
    if val_losses[-1] < model_performances["epochs"]["loss"]:
      model_performances["epochs"]["name"] = epochs
      model_performances["epochs"]["loss"] = val_losses[-1]

  plt.figure(figsize=(8, 5)) 
  plt.plot(num_epochs, val_results, marker='o', linestyle='-', color='blue')
  plt.xlabel("Number of Epochs")
  plt.ylabel("Validation Performance (RMSE)")
  plt.title("Validation Performance of Different Number of Epochs")
  plt.show()
  # plt.savefig("epochs.png")
  # plt.clf() 

  # d) Activation Functions
  activations = {
    "ReLU": torch.nn.ReLU(),
    "Tanh": torch.nn.Tanh(),
    "Sigmoid": torch.nn.Sigmoid(),
    "LeakyReLU": torch.nn.LeakyReLU(),
    "GELU": torch.nn.GELU()
  }
  val_results = {}

  for name, activation in activations.items():
    model, _, val_losses = train([64, 32], activation, 200)
    val_results[name] = val_losses[-1]
    if val_losses[-1] < model_performances["activation"]["loss"]:
      model_performances["activation"]["name"] = activation
      model_performances["activation"]["loss"] = val_losses[-1]

  names = list(val_results.keys())
  values = list(val_results.values())

  plt.figure(figsize=(8, 5)) 
  plt.bar(names, values, color='blue')
  plt.xlabel("Activation Function")
  plt.ylabel("Validation Performance (RMSE)")
  plt.title("Validation Performance of Different Activation Functions")
  plt.show()
  # plt.savefig("activations.png")
  # plt.clf() 

  # e) Models with overall best performance
  best_models = {}
  print(f"QUESTION 4: Best performing models")
  for key, value in model_performances.items():
    print(f"{key}: {value["name"]}")
    best_models[key] = value["name"]

  best_model = {}
  min_loss = math.inf
  print(f"\nQUESTION 4: Best model out of all experiments")
  for key, value in model_performances.items():
    if value["loss"] < min_loss:
      best_model["type"] = key
      best_model["name"] = value["name"]
      best_model["loss"] = value["loss"]
      min_loss = value["loss"]

  final_model = None
  match best_model["type"]:
    case "neurons":
      final_model, _, _ = train([best_model["name"]], torch.nn.ReLU(), 200)
    case "layers":
      best_model, _, _ = train(best_model["name"], torch.nn.ReLU(), 200)
    case "epochs":
      final_model, _, _ = train([64, 32], torch.nn.ReLU(), best_model["name"])
    case "activation":
      final_model, _, _ = train([64, 32], best_model["name"], 200)
    case _:
      final_model = None

  
  # Best model (from experiments)
  print(f"\nBest Model Performance (single model from experiments): {best_model["type"]} = {best_model["name"]}")

  loss_func = torch.nn.MSELoss()
  final_model.eval()
  with torch.no_grad():
    train_rmse = torch.sqrt(loss_func(final_model(X_train), Y_train))
    val_rmse   = torch.sqrt(loss_func(final_model(X_val), Y_val))
    test_rmse  = torch.sqrt(loss_func(final_model(X_test), Y_test))

  print("Train RMSE:", train_rmse.item())
  print("Validation RMSE:", val_rmse.item())
  print("Test RMSE:", test_rmse.item())

  # Overall best model (combining experiments)
  print(f"\nBest Overall Model Performance (combining models from experiments):")
  overall_final_model, _, _ = train(best_models["layers"], best_models["activation"], best_models["epochs"])

  loss_func = torch.nn.MSELoss()
  overall_final_model.eval()

  with torch.no_grad():
    train_rmse = torch.sqrt(loss_func(overall_final_model(X_train), Y_train))
    val_rmse   = torch.sqrt(loss_func(overall_final_model(X_val), Y_val))
    test_rmse  = torch.sqrt(loss_func(overall_final_model(X_test), Y_test))

  print(f"Neurons: {best_models["neurons"]}")
  print(f"Layers: {best_models["layers"]}")
  print(f"Epochs: {best_models["epochs"]}")
  print(f"Activation Function: {best_models["activation"]}")
  print("Train RMSE:", train_rmse.item())
  print("Validation RMSE:", val_rmse.item())
  print("Test RMSE:", test_rmse.item())

if __name__ == "__main__":

  # Question 2
  print(f"QUESTION 2: multitask_training")
  filepath = '/home/andrew/Documents/cu/w26/comp4107/a2/multitask_data.csv'
  eval_filepath = '/home/andrew/Documents/cu/w26/comp4107/a2/multitask_data_eval.csv'
  model = multitask_training(filepath) 
  print(model)
  print(f"multitask_training evaluation")
  multitask_evaluation(model, eval_filepath)

  # Question 3
  print(f"\nQUESTION 3: mlb_position_player_salary")
  filepath = '/home/andrew/Documents/cu/w26/comp4107/a2/baseball.txt'
  model, validation_performance = mlb_position_player_salary(filepath)
  print(validation_performance)

  # Question 4
  filepath = '/home/andrew/Documents/cu/w26/comp4107/a2/baseball.txt'
  print(f"\nQUESTION 3: mlb_experiments")
  mlb_experiments(filepath)