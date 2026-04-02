# Name this file assignment5.py when you submit
import torch
import numpy as np
import pandas as pd
import re
from collections import Counter
from torch.utils.data import DataLoader
import random

class FinancialNewsDataset(torch.utils.data.Dataset):
  """ PyTorch dataset for the financial news dataset """
  
  mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
  }
  
  shared_vocab_to_int = None
  shared_max_len = None

  def __init__(self, filepath, rows):
    """
      Parameters:
        filepath: is a full file path to the file containing the data
        rows: is the rows from the file to use
    """
    
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
   
    # Get word counts 
    if FinancialNewsDataset.shared_vocab_to_int is None:
      all_words = [word for headline in self.data['headline'] for word in headline.split()]
      counts = Counter(all_words)
      vocab = sorted(counts, key=counts.get, reverse=True)

      vocab_to_int = {word: ii+1 for ii, word in enumerate(vocab)}
      vocab_to_int["<PAD>"] = 0
      vocab_to_int["<UNK>"] = len(vocab_to_int) # append to end
      FinancialNewsDataset.shared_vocab_to_int = vocab_to_int
    
    self.vocab_to_int = FinancialNewsDataset.shared_vocab_to_int
    self.vocab_size = len(self.vocab_to_int)
    
    # Tokenize
    def _tokenize(text):
      return [self.vocab_to_int.get(word, self.vocab_to_int["<UNK>"]) for word in text.split()]
    self.data['tokenized'] = self.data['headline'].apply(_tokenize)
    
    # Define max_len of a sequence
    if FinancialNewsDataset.shared_max_len is None:
      FinancialNewsDataset.shared_max_len = int(
        self.data['tokenized'].apply(len).quantile(0.95)
      )
    self.max_len = FinancialNewsDataset.shared_max_len
     
    # Pad/crop sequences
    self.data['padded'] = self.data['tokenized'].apply(
      lambda x: self._pad_sequence(x, self.max_len)
    )
    
    # Filter rows
    if rows is not None:
      self.data = self.data.iloc[rows].reset_index(drop=True)
    
    # TODO: Data augmentation?
    
    # Return nothing    

  def __len__(self):
    """ num_samples is the total number of samples in the dataset """
    num_samples = len(self.data)
    return num_samples


  def __getitem__(self, index):
    """
      Parameters:
        index: is the index of the sample to be retrieved
      Returns:
        x: is one sample of data (i.e. one headline)
        y: is the label associated with the sample (i.e. its sentiment)
    """
    
    x = self.data.loc[index, 'padded'] 
    x = torch.tensor(x, dtype=torch.long)
    
    # y = self._one_hot_label(self.data.loc[index, 'sentiment'])
    y = self.data.loc[index, 'label_index']
    y = torch.tensor(y, dtype=torch.long)
    
    return x, y
  
  def _clean_text(self, text):
    # Lowercase
    text = text.lower()
    
    # Replace hyphens with space 
    text = re.sub(r'-', ' ', text)
    
    # Keep decimal numbers
    text = re.sub(r'(\d)\.(\d)', r'\1.\2', text)

    # Normalize percentages
    text = re.sub(r'(\d+)\s*%', r'\1 percent', text)

    # Normalize financial units
    text = re.sub(r'\bmn\b', 'million', text)
    text = re.sub(r'\bbn\b', 'billion', text)
    
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
  
  def _one_hot_label(self, label):
    index = self.mapping[label]
    return np.eye(len(self.mapping))[index]
  
  def _pad_sequence(self, seq, max_len):
    if len(seq) > max_len:
        return seq[:max_len]  # truncate
    return seq + [0] * (max_len - len(seq))


def financial_news_rnn_model(data_filepath, training_rows, validation_rows):
  """
    A function that creates an rnn model for the financial news dataset
    Parameters:
      data_filepath: is a full file path to the file containing the data
      training_rows: is the rows from the dataset to use for training
      validation_rows: is the rows from the dataset to use for validation
    Returns:
      model: is a trained rnn model for this task
      training_performance: is the performance of the model on the training set
      validation_performance: is the performance of the model on the validation set
  """
  
  train_dataset = FinancialNewsDataset(data_filepath, training_rows)
  val_dataset = FinancialNewsDataset(data_filepath, validation_rows)
  
  vocab_size = train_dataset.vocab_size
  print(f"Vocab size: {vocab_size}")
  
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=64)
  
  class RNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=64, output_dim=3):
      super().__init__()
      self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
      self.rnn = torch.nn.GRU(embed_dim, hidden_dim, batch_first=True)
      self.dropout = torch.nn.Dropout(0.5)
      self.fc = torch.nn.Linear(hidden_dim, output_dim)
      
    def forward(self, x):
      x = self.embedding(x)
      out, _ = self.rnn(x)
      out = self.dropout(out.mean(dim=1))
      out = self.fc(out)
      return out
    
  def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for x_batch, y_batch in loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
      
    return total_loss / len(loader)

  
  def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
      for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item()
    
        preds = outputs.argmax(dim=1) 
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(loader), correct / total
   
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = RNN(vocab_size=vocab_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = torch.nn.CrossEntropyLoss()
  
  model.to(device)
  
  epochs = 50
  all_train_losses = []
  all_val_losses = []
  for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    all_train_losses.append(train_loss)
    all_val_losses.append(val_loss)
    
    print(f"EPOCH {epoch + 1}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}, Val accuracy={val_acc:.4f}")
    
  training_performance = np.mean(all_train_losses)
  validation_performance = np.mean(all_val_losses)
  
  return model, training_performance, validation_performance


# 
def financial_news_attention_model(data_filepath, training_rows, validation_rows):
  """
    A function that creates an attention-based model for the financial news dataset
    Parameters:
      data_filepath: is a full file path to the file containing the data
      training_rows: is the rows from the dataset to use for training
      validation_rows: is the rows from the dataset to use for validation
    Returns:
      model: is a trained attention-based model for this task
      training_performance: is the performance of the model on the training set
      validation_performance: is the performance of the model on the validation set 
  """
  # class RNN(torch.nn.Module):
     
  #   def __init__(self, vocab_size, embed_dim=50, hidden_dim=64, output_dim=3):
  #     super().__init__()
  #     self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
  #     self.rnn = torch.nn.GRU(embed_dim, hidden_dim, batch_first=True)
  #     self.attention = torch.nn.Linear(hidden_dim, 1)
  #     self.dropout = torch.nn.Dropout(0.5)
  #     self.fc = torch.nn.Linear(hidden_dim, output_dim)
      
  #   def forward(self, x):
  #     x = self.embedding(x)
  #     out, _ = self.rnn(x)
      
  #     # Add attention layer from output of RNN before passing to output layer
      
  #     # Calculate scores for each hidden state (seq_len, hidden_dim)
  #     # (batch_size, seq_len, 1)
  #     scores = self.attention(out) 
  #     scores = scores.squeeze(-1) # (batch_size, seq_len)
      
  #     # Get the weights alpha_{i,j} from scores using softmax
  #     weights = torch.softmax(scores, dim=1)
  #     context = torch.bmm(weights.unsqueeze(1), out) # (batch_size, 1, hidden_dim )
  #     context = context.squeeze(1) # (batch_size, hidden_dim)
      
  #     out = self.dropout(context)
  #     out = self.fc(out)
  #     return out

  
  return model, training_performance, validation_performance


if __name__ == "__main__":
  
  filepath = "/home/andrew/w26/comp4107/a5/all-data.csv"
    
  num_rows = FinancialNewsDataset(filepath, None).__len__()
  all_rows = list(range(num_rows))
  random.shuffle(all_rows)
  
  # Split into 75% for train and 20% for validation, 5% for visualization
  split = int(num_rows * 0.75) 
  num_rows2 = num_rows - split
  split2 = int(num_rows2 * 0.95) 
  print(split, split2)
  train_rows = all_rows[:split]
  val_rows = all_rows[split:split + split2]
  val_viz_rows = all_rows[split + split2:]
  
  # Sanity check 
  '''
  ds = FinancialNewsDataset(filepath, [1,2,3])
  print(ds[0][0])
  print(len(ds[0][0]))
  print(ds[1][0])
  print(len(ds[1][0]))
  print(ds[2][0])
  print(len(ds[2][0]))
  '''
  
  # Evaluate RNN
  model, train_performance, val_performance = financial_news_rnn_model(filepath, train_rows, val_rows)
  print(f"Final train performance: {train_performance}")
  print(f"Final val performance: {val_performance}")
  
  def visualize_preds(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    print("visualize preds")
    
    with torch.no_grad():
      for x_batch, y_batch in loader:
        print(x_batch)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy()) # predictions
        all_labels.extend(y_batch.cpu().numpy()) # labels
        
    index_to_sentiment = {0: "negative", 1: "neutral", 2: "positive"}
    pred_labels = [index_to_sentiment[p] for p in all_preds]
    true_labels = [index_to_sentiment[t] for t in all_labels]

    for pred, true in zip(pred_labels, true_labels):
      print(f"Predicted: {pred}, True: {true}")
  
  ds = FinancialNewsDataset(filepath, val_viz_rows)
  val_viz_loader = DataLoader(ds, batch_size=64) 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  visualize_preds(model, val_viz_loader, device)
        
 