# Name this file assignment3.py when you submit
import torch

# PyTorch dataset for the UWaveGestureLibrary dataset
class UWaveGestureLibraryDataset(torch.utils.data.Dataset):

  def __init__(self, dataset_filepath):
    # dataset_filepath is the full path to a file containing data

    # Return nothing    

  def __len__(self):
    # num_samples is the total number of samples in the dataset
    return num_samples


  def __getitem__(self, index):
    # index is the index of the sample to be retrieved
    
    # x is one sample of data
    # y is the label associated with the sample
    return x, y


# A function that creates a cnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_cnn_model(training_data_filepath):
  # training_data_filepath is the full path to a file containing the training data

  # model is a trained cnn model to predict which class a sequence corresponds to
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance, validation_performance


# A function that creates an rnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_rnn_model(training_data_filepath):
  # training_data_filepath is the full path to a file containing the training data

  # model is a trained rnn model to predict which class a sequence corresponds to
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance, validation_performance
