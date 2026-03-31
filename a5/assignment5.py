# Name this file assignment5.py when you submit
import torch
import numpy as np
import pandas as pd
from string import punctuation
import re
from collections import Counter

# PyTorch dataset for the financial news dataset
class FinancialNewsDataset(torch.utils.data.Dataset):
  
  mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
  }

  def __init__(self, filepath, rows):
    # filepath is a full file path to the file containing the data
    # rows is the rows from the file to use
    self.path = filepath
    self.num_classes = 3
    
    # Read CSV safely
    try:
      self.data = pd.read_csv(
        filepath,
        header=None,
        names=["sentiment", "headline"],
        quotechar='"',      # handle quotes around text
        encoding='utf-8',   # first try utf-8
      )
    except UnicodeDecodeError:
      # fallback if utf-8 fails
      print('utf-8 failed')
      self.data = pd.read_csv(
        filepath,
        header=None,
        names=["sentiment", "headline"],
        quotechar='"',
        encoding='latin1' 
      )
      
    # Clean headlines
    self.data['headline'] = self.data['headline'].apply(self._clean_text)
    
    # Map labels to index
    self.data['label_index'] = self.data['sentiment'].map(self.mapping)
    
    # Tokenize
    all_words = [word for headline in self.data['headline'] for word in headline.split()]
    counts = Counter(all_words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    def _tokenize(text):
      return [vocab_to_int[word] for word in text.split()] 
    self.data['tokenized'] = self.data['headline'].apply(_tokenize)
    
    # Filter rows
    if rows is not None:
      self.data = self.data.iloc[rows].reset_index(drop=True)
    
    # TODO: Data augmentation?
    
    # Return nothing    

  def __len__(self):
    # num_samples is the total number of samples in the dataset
    num_samples = len(self.data)
    return num_samples


  def __getitem__(self, index):
    # index is the index of the sample to be retrieved
    
    # x is one sample of data (i.e. one headline)
    # y is the label associated with the sample (i.e. its sentiment)
    x = self.data.loc[index, 'tokenized'] 
    x = torch.tensor(x, dtype=torch.float32)
    
    y = self._one_hot_label(self.data.loc[index, 'sentiment'])
    y = torch.tensor(y, dtype=torch.float32)
    
    return x, y
  
  def _clean_text(self, text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation 
    text = ''.join([c for c in text if c not in punctuation])
    
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
  
  def _one_hot_label(self, label):
    index = self.mapping[label]
    return np.eye(len(self.mapping))[index]
  
    


# A function that creates an rnn model for the financial news dataset
def financial_news_rnn_model(data_filepath, training_rows, validation_rows):
  # data_filepath is a full file path to the file containing the data
  # training_rows is the rows from the dataset to use for training
  # validation_rows is the rows from the dataset to use for validation

  # model is a trained rnn model for this task
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance, validation_performance


# A function that creates an attention-based model for the financial news dataset
def financial_news_attention_model(data_filepath, training_rows, validation_rows):
  # data_filepath is a full file path to the file containing the data
  # training_rows is the rows from the dataset to use for training
  # validation_rows is the rows from the dataset to use for validation

  # model is a trained attention-based model for this task
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance, validation_performance


if __name__ == "__main__":
  
  filepath = "/home/andrew/w26/comp4107/a5/all-data.csv"
  ds = FinancialNewsDataset(filepath, [1,2,3])
  print(ds.__getitem__(0))