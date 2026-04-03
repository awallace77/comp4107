# Name this file assignment5.py when you submit
import torch
import numpy as np
import pandas as pd
import re
from collections import Counter
from torch.utils.data import DataLoader
import random
import copy

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
      # top_words = 6000
      # vocab_to_int = {word: ii+1 for ii, word in enumerate(vocab[:top_words])}
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
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=32, output_dim=3):
      super().__init__()
      self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
      self.embedding_dropout = torch.nn.Dropout(0.4)
      self.rnn = torch.nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=2, bidirectional=True, dropout=0.3)
      self.dropout = torch.nn.Dropout(0.6)
      self.fc = torch.nn.Linear(hidden_dim * 2, output_dim) # * 2 b/c bidirectional
      
    def forward(self, x):
      x = self.embedding(x)
      x = self.embedding_dropout(x)
      out, _ = self.rnn(x)
      out = self.dropout(out.mean(dim=1))
      out = self.fc(out)
      return out
    
  def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x_batch, y_batch in loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
      
      preds = outputs.argmax(dim=1)
      correct += (preds == y_batch).sum().item()
      total += y_batch.size(0)
     
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

  
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
  best_model = copy.deepcopy(model.state_dict())
  best_val_loss = float('inf')
  
  epochs = 50
  
  all_train_losses = []
  all_train_acc = []
  all_val_losses = []
  all_val_acc = []
  best_model_val_loss = float('inf')
  best_model_val_acc = float('inf')
  
  for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    all_train_losses.append(train_loss)
    all_train_acc.append(train_acc)
    all_val_losses.append(val_loss) 
    all_val_acc.append(val_acc)
    
    # Save best model
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_model = copy.deepcopy(model.state_dict())
      best_model_val_loss = val_loss
      best_model_val_acc = val_acc
    
    print(f"EPOCH {epoch + 1}:")
    print(f"  Train loss = {train_loss:.4f} | Train accuracy = {train_acc:.4f}")
    print(f"  Val   loss = {val_loss:.4f} | Val   accuracy = {val_acc:.4f}")
    
  training_performance = (np.min(all_train_losses), np.max(all_train_acc))
  validation_performance = (best_model_val_loss, best_model_val_acc)
  model.load_state_dict(best_model) # restore the best model
  
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
  
  class AdditiveAttention(torch.nn.Module):
    
    def __init__(self, hidden_dim):
      super().__init__()
      self.W_h = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
      self.W_q = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
      self.v = torch.nn.Linear(hidden_dim, 1, bias=False)
      
    def forward(self, h, q):
      """
        h is the hidden state (batch_size, seq_len, hidden_dim)
        q is the query (e.g., one hidden state) (batch_size, hidden_dim)
      """
      q = q.unsqueeze(1) # (batch_size, 1, hidden_dim)
      
      # Additive scoring
      score = self.v(torch.tanh(self.W_h(h) + self.W_q(q)))
      alphas = torch.softmax(score, dim=1) # (batch_size, seq_len, 1)
      context = (alphas * h).sum(dim=1)  # (batch_size, hidden_dim)
      
      # Context vector 
      return context
      
      
  class AttentionRNN(torch.nn.Module):
     
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=32, output_dim=3, n_attention=2):
      super().__init__()
      
      # Embedding layer
      self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
      self.embedding_dropout = torch.nn.Dropout(0.4)
      
      # RNN
      self.rnn = torch.nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=2, bidirectional=True, dropout=0.3)
      
      # Attention layers
      self.attentions = torch.nn.ModuleList(
        [AdditiveAttention(hidden_dim * 2) for _ in range(n_attention)]
      )
      
      # Learned queries
      self.query_layers = torch.nn.ModuleList(
        [torch.nn.Linear(hidden_dim * 2, hidden_dim * 2) for _ in range(n_attention)]
      )
      
      # Dropout
      self.dropout = torch.nn.Dropout(0.6)
      
      # Fully connected layer
      self.fc = torch.nn.Linear(hidden_dim * 2 * n_attention, output_dim)
      
    def forward(self, x):
      x = self.embedding(x)
      x = self.embedding_dropout(x)
      h_enc, h_T = self.rnn(x)
      h_enc = self.dropout(h_enc) 
     
      h_forward = h_T[-2]
      h_backward = h_T[-1]
      h_last = torch.cat((h_forward, h_backward), dim=1)
      # h_last = h_T.squeeze(0)
      
      # Pass through attention layers 
      context_vectors = []
      for i, attention in enumerate(self.attentions):
        # Use a different query per attention layer via
        query = torch.tanh(self.query_layers[i](h_last))
        context = attention(h_enc, query)
        context_vectors.append(context)
      
      # Concat all context vectors
      context = torch.cat(context_vectors, dim=1) 
      context = self.dropout(context)
      
      out = self.fc(context)
      return out
    
  def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x_batch, y_batch in loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
      
      preds = outputs.argmax(dim=1)
      correct += (preds == y_batch).sum().item()
      total += y_batch.size(0)
     
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
      
    return avg_loss, accuracy

  
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
     
     
  train_dataset = FinancialNewsDataset(data_filepath, training_rows)
  val_dataset = FinancialNewsDataset(data_filepath, validation_rows)
  
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=64)
  
  vocab_size = train_dataset.vocab_size
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = AttentionRNN(vocab_size=vocab_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = torch.nn.CrossEntropyLoss()
  
  model.to(device)
  best_model = copy.deepcopy(model.state_dict())
  best_val_loss = float('inf')
  
  epochs = 50
  
  all_train_losses = []
  all_train_acc = []
  all_val_losses = []
  all_val_acc= []
  
  best_model_val_loss = float('inf')
  best_model_val_acc = float('inf')
  
  for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    all_train_losses.append(train_loss)
    all_train_acc.append(train_acc)
    all_val_losses.append(val_loss)
    all_val_acc.append(val_acc)
    
    # Save best model
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      print(f"Best new model found")
      best_model = copy.deepcopy(model.state_dict())
      best_model_val_loss = val_loss
      best_model_val_acc = val_acc

    print(f"EPOCH {epoch + 1}:")
    print(f"  Train loss = {train_loss:.4f} | Train accuracy = {train_acc:.4f}")
    print(f"  Val   loss = {val_loss:.4f} | Val   accuracy = {val_acc:.4f}")
    
  training_performance = (np.min(all_train_losses), np.max(all_train_acc))
  validation_performance = (best_model_val_loss, best_model_val_acc)
  model.load_state_dict(best_model) # restore the best model
  
  return model, training_performance, validation_performance


if __name__ == "__main__":
  
  filepath = "/home/andrew/w26/comp4107/a5/all-data.csv"
    
  num_rows = FinancialNewsDataset(filepath, None).__len__()
  all_rows = list(range(num_rows))
  random.shuffle(all_rows)
  
  # Split into 75% for train and 25% for validation
  split = int(num_rows * 0.75)
  train_rows = all_rows[:split]
  val_rows = all_rows[split:]
  
  ## Question 2: Evaluate RNN
  # model, train_performance, val_performance = financial_news_rnn_model(filepath, train_rows, val_rows)
  # print(f"RNN: Final train performance (CE Loss) = {train_performance[0]}, Final train accuracy = {train_performance[1]}")
  # print(f"RNN: Final val performance (CE Loss)   = {val_performance[0]}  , Final val accuracy   = {val_performance[1]}")
 
  ## Question 3: Evaluate Attention model
  model, train_performance, val_performance = financial_news_attention_model(filepath, train_rows, val_rows)
  print(f"ATTENTION: Final train performance (CE Loss) = {train_performance[0]:.4f}, Final train accuracy = {train_performance[1]:.4f}")
  print(f"ATTENTION: Final val performance (CE Loss)   = {val_performance[0]:.4f},   Final val accuracy   = {val_performance[1]:.4f}")
  
  def visualize():
    split = int(num_rows * 0.25) 
    val_viz_rows = all_rows[:split]
    
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
 