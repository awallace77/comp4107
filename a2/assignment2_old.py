# Name this file assignment2.py when you submit
import numpy
import torch

# A function that implements a pytorch model following the provided description
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


# A function that implements training following the provided description
def multitask_training(data_filepath):
  """
    Trains the mutlitask network on training data
      - A loss function, which is computed as a sum of categorical cross-entropy losses for each of the three-class classification tasks.
      - An optimizer using stochastic gradient descent with a cosine learning rate schedule.
      - Each row in the file refers to an instance (i.e. an MLB position player).
  """
  num_epochs = 100
  batch_size = 4

  data = numpy.loadtxt(data_filepath, delimiter=",")
  batches_per_epoch = int(data.shape[0] / batch_size)

  multitask_network = MultitaskNetwork()

  # Define loss function(s) here
  loss_func1 = torch.nn.CrossEntropyLoss()
  loss_func2 = torch.nn.CrossEntropyLoss()

  # Define optimizer here
  lr = 0.001
  optimizer = torch.optim.SGD(multitask_network.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


  for epoch in range(num_epochs):
    for batch_index in range(batches_per_epoch):
      x = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 6:9], dtype=torch.float32)
      y_a = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 0:3], dtype=torch.float32)
      y_b = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 3:6], dtype=torch.float32)

      y_pred_a, y_pred_b = multitask_network(x)

      # Compute loss here
      loss = loss_func1(torch.squeeze(y_pred_a), y_a) + loss_func2(torch.squeeze(y_pred_b), y_b)

      # Compute gradients here
      optimizer.zero_grad()
      loss.backward()

      # Update parameters according to SGD with learning rate schedule here
      optimizer.step()
      scheduler.step()


  # A trained torch.nn.Module object
  return multitask_network


def mlb_position_player_salary(filepath):
  """
    A function that creates a pytorch model to predict the salary of an MLB position player
    
    :param filepath: is the path to an csv file containing the dataset
  """

  class MLBNetwork(torch.nn.Module):
    def __init__(self):
      super().__init__()

      self.relu = torch.nn.ReLU()
      self.layer1 = torch.nn.Linear(16, 32)
      self.layer2 = torch.nn.Linear(32, 16)
      self.layer3 = torch.nn.Linear(16, 1)

    def forward(self, x):

      x = self.layer1(x)
      x = self.relu(x)
      x = self.layer2(x)
      x = self.relu(x)
      x = self.layer3(x)

      return x
  
  num_epochs = 200
  batch_size = 8

  data = numpy.loadtxt(filepath, delimiter=",", skiprows=1)

  model = MLBNetwork()

  # Define loss function(s) here
  loss_func = torch.nn.MSELoss()
  
  # model is a trained pytorch model for predicting the salary of an MLB position player
  
  lr = 0.01
  optimizer = torch.optim.Adam(model .parameters(), lr=lr)
  # optimizer = torch.optim.AdamW(model .parameters(), lr=lr, weight_decay=1e-4)

  perm = numpy.random.permutation(data.shape[0]) 
  data = data[perm]

  train_size = int(0.8 * data.shape[0])
  train_data = data[:train_size, :]
  test_data = data[train_size:, :]
  
  X_train = train_data[:, 1:]
  Y_train = train_data[:, 0:1]
  X_test = test_data[:, 1:]
  Y_test = test_data[:, 0:1]
  
  batches_per_epoch = int(train_data.shape[0] / batch_size)

  model.train()
  for epoch in range(num_epochs):
    
    epoch_loss = 0.0

    for batch_index in range(batches_per_epoch):
      
      x = torch.as_tensor(X_train[batch_index * batch_size:(batch_index + 1) * batch_size, :], dtype=torch.float32)
      y = torch.as_tensor(Y_train[batch_index * batch_size:(batch_index + 1) * batch_size, :], dtype=torch.float32)

      # Prediction
      y_pred = model(x)

      # Loss
      loss = loss_func(torch.squeeze(y_pred), torch.squeeze(y))

      if torch.isnan(loss):
        print(f"Found NANA for {torch.squeeze(y_pred)} and {torch.squeeze(y)}")
      else:
        epoch_loss += loss.item()

      # Backpropagation 
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
    print(f"Epoch: {epoch} || Loss: {epoch_loss / (batch_size * batches_per_epoch)}")


  # validation_performance is the performance of the model on a validation set

  model.eval()
  y_test = torch.squeeze(torch.as_tensor(Y_test, dtype=torch.float32))
  y_test_pred = torch.squeeze(model(torch.as_tensor(X_test, dtype=torch.float32)))

  loss_func = torch.nn.MSELoss()
  loss = loss_func(y_test_pred, y_test)
  validation_performance = torch.sqrt(loss)

  print(f"PREDICTED LABEL : {y_test_pred}")
  print(f"ACTUAL LABEL    : {y_test}")
  print(f"LOSS            : {validation_performance}")

  return model, validation_performance


if __name__ == "__main__":

  # Question 2
  print(f"QUESTION 2: multitask_training")
  filepath = '/home/andrew/Documents/cu/w26/comp4107/a2/multitask_data.csv'
  model = multitask_training(filepath) 
  print(model)
  print("")

  # Question 3
  print(f"QUESTION 3: mlb_position_player_salary")
  filepath = '/home/andrew/Documents/cu/w26/comp4107/a2/baseball.txt'
  model, validation_performance = mlb_position_player_salary(filepath)
  print("")
