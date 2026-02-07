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


# A function that creates a pytorch model to predict the salary of an MLB position player
def mlb_position_player_salary(filepath):
  # filepath is the path to an csv file containing the dataset

  # model is a trained pytorch model for predicting the salary of an MLB position player
  # validation_performance is the performance of the model on a validation set
  return model, validation_performance


if __name__ == "__main__":

  file_path = './multitask_data.csv'
  model = multitask_training(file_path) 

  print(model)