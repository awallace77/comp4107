# Name this file assignment3.py when you submit
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt

# PyTorch dataset for the UWaveGestureLibrary dataset
class UWaveGestureLibraryDataset(torch.utils.data.Dataset):

  def __init__(self, dataset_filepath):
    # dataset_filepath is the full path to a file containing data
    self.path = dataset_filepath
    self.numClasses = 8

    inputList = []
    outputList = []

    with open(self.path, "r", encoding="utf-8") as f:
      for line in f:
        parts = line.split(':')
        x = [float(v) for v in parts[0].split(",")]
        y = [float(v) for v in parts[1].split(",")]
        z = [float(v) for v in parts[2].split(",")]
        label = int(float(parts[3].strip()))

        sample = torch.tensor([x, y, z], dtype=torch.float32)

        y_index = label - 1  # Reduce the label by one for easier one-hot classification

        inputList.append(sample)
        outputList.append(y_index)

    self.X = torch.stack(inputList)               # Shape: (N, 3, 315)
    self.y = torch.tensor(outputList, dtype=torch.long)  # Shape: (N,)

    # Normalize
    # self.mean = self.X.mean(dim=(0, 2), keepdim=True) 
    # self.std = self.X.std(dim=(0, 2), keepdim=True)
    # self.X = (self.X - self.mean) / (self.std + 1e-6)

  def __len__(self):
    # num_samples is the total number of samples in the dataset
    return self.X.shape[0]


  def __getitem__(self, index):
    # index is the index of the sample to be retrieved
    
    # x is one sample of data
    # y is the label associated with the sample
    x = self.X[index]
    y_idx = self.y[index]
    y_onehot = torch.nn.functional.one_hot(y_idx, num_classes=self.numClasses).to(torch.float32)
    return x, y_onehot



def trainValidationSplit(dataset):
  idx = np.arange(len(dataset))
  trainIndex, valIndex = train_test_split(
    idx,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=dataset.y.numpy()
  )

  trainDataset = Subset(dataset, trainIndex)
  valDataset   = Subset(dataset, valIndex)

  trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=True)
  valLoader   = DataLoader(valDataset, batch_size=256, shuffle=False)
  return trainLoader, valLoader


# A function that creates a cnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_cnn_model(training_data_filepath):
  # training_data_filepath is the full path to a file containing the training data
  ds = UWaveGestureLibraryDataset(training_data_filepath)
  # model is a trained cnn model to predict which class a sequence corresponds to
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set

  trainLoader, valLoader = trainValidationSplit(ds)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = torch.nn.Sequential(
        # (B, 3, 315) -> (B, 32, 158)
        torch.nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
        torch.nn.ReLU(),

        # (B, 32, 158) -> (B, 64, 79)
        torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
        torch.nn.ReLU(),

        # (B, 64, 79) -> (B, 128, 40)
        torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),

        # (B, 128, 40) -> (B, 128, 40)
        torch.nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),

        torch.nn.Flatten(),                 # (B, 128*40)
        torch.nn.Linear(128 * 40, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, ds.numClasses)    # (B, 8)
    ).to(device)
  
  criterion = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

  def run_epoch(loader, training: bool):
      model.train() if training else model.eval()

      total_loss = 0.0
      correct = 0
      total = 0

      for xb, y_onehot in loader:
          xb = xb.to(device)              # (B, 3, 315)
          y_onehot = y_onehot.to(device)  # (B, 8)

          if training:
              optimizer.zero_grad()

          with torch.set_grad_enabled(training):
              logits = model(xb)          # (B, 8)
              loss = criterion(logits, y_onehot)

              if training:
                  loss.backward()
                  optimizer.step()

          total_loss += loss.item() * xb.size(0)

          preds = logits.argmax(dim=1)
          true  = y_onehot.argmax(dim=1)
          correct += (preds == true).sum().item()
          total += xb.size(0)

      return total_loss / total, correct / total

  # -------- train loop ----------
  epochs = 100
  bestValAccuracy = -1
  bestState = None

  for epoch in range(1, epochs + 1):
      trainLoss, trainAcc = run_epoch(trainLoader, training=True)
      valLoss, valAcc = run_epoch(valLoader, training=False)

      if valAcc > bestValAccuracy:
          bestValAccuracy = valAcc
          bestState = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

      print(f"Epoch {epoch:02d} | "
            f"train loss {trainLoss:.4f} acc {trainAcc:.3f} | "
            f"val loss {valLoss:.4f} acc {valAcc:.3f}")

  if bestState is not None:
      model.load_state_dict(bestState)

  _, _ = run_epoch(trainLoader, training=False)
  training_performance = {"loss": float(trainLoss), "accuracy": float(trainAcc)}
  validation_performance = {"loss": float(valLoss), "accuracy": float(bestValAccuracy)}

  return model, training_performance, validation_performance


# A function that creates an rnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_rnn_model_gru(training_data_filepath):
    # training_data_filepath is the full path to a file containing the training data

    ds = UWaveGestureLibraryDataset(training_data_filepath)
    trainLoader, valLoader = trainValidationSplit(ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class RNNModel(torch.nn.Module):
      def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=8):
        super().__init__()

        self.rnn = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5, 
            bidirectional=True # ----
        )

        rnn_out_size = hidden_size * 4

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_out_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )

      def forward(self, x):
        # x: (B, 315, 3)
        output, hidden = self.rnn(x)  # output: (B, 315, hidden*2)

        mean_pool = output.mean(dim=1)          # (B, hidden*2)
        max_pool  = output.max(dim=1).values    # (B, hidden*2)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, hidden*4)

        return self.classifier(pooled)

    model = RNNModel(
        input_size=3,
        hidden_size=256,
        num_layers=2,
        num_classes=ds.numClasses
    ).to(device)

    epochs = 50
    bestValAccuracy = -1
    bestState = None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def run_epoch(loader, training):
        model.train() if training else model.eval()

        totalLoss = 0.0
        correct = 0
        total = 0

        for xb, y_onehot in loader:
            xb = xb.to(device)              # (B, 3, 315)
            y_onehot = y_onehot.to(device)  # (B, 8)

            xb = xb.transpose(1, 2)         # (B, 315, 3)

            if training:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                logits = model(xb)          # (B, 8)
                loss = criterion(logits, y_onehot)

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            totalLoss += loss.item() * xb.size(0)

            preds = logits.argmax(dim=1)
            true = y_onehot.argmax(dim=1)
            correct += (preds == true).sum().item()
            total += xb.size(0)

        return totalLoss / total, correct / total

    

    for epoch in range(1, epochs + 1):
      trainLoss, trainAcc = run_epoch(trainLoader, training=True)
      valLoss, valAcc = run_epoch(valLoader, training=False)
      scheduler.step()

      if valAcc > bestValAccuracy:
        bestValAccuracy = valAcc
        bestState = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

      print(f"Epoch {epoch:02d} | "
        f"train loss {trainLoss:.4f} acc {trainAcc:.3f} | "
        f"val loss {valLoss:.4f} acc {valAcc:.3f}")

    if bestState is not None:
      model.load_state_dict(bestState)

    bestTrainLoss, bestTrainAcc = run_epoch(trainLoader, training=False)
    training_performance = {"loss": float(bestTrainLoss), "accuracy": float(bestTrainAcc)}

    validation_performance = {
        "loss": float(valLoss),
        "accuracy": float(bestValAccuracy)
    }

    return model, training_performance, validation_performance


class RNNModel(torch.nn.Module):
      def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=8):
        super().__init__()

        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5, 
            bidirectional=True
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 4, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )

      def forward(self, x):
        # x: (B, 315, 3)
        output, hidden = self.rnn(x)  # output: (B, 315, hidden*2)
        mean_pool = output.mean(dim=1)          # (B, hidden*2)
        max_pool  = output.max(dim=1).values    # (B, hidden*2)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, hidden*4)

        return self.classifier(pooled)
    
def train_rnn_model(training_data_filepath, num_epochs, num_hidden_states, num_layers=2, log=False):
    # Data initialization
    ds = UWaveGestureLibraryDataset(training_data_filepath)
    trainLoader, valLoader = trainValidationSplit(ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Define model 
    model = RNNModel(
        input_size=3,
        hidden_size=num_hidden_states,
        num_layers=num_layers,
        num_classes=ds.numClasses
    ).to(device)

    epochs = num_epochs
    bestValAccuracy = -1
    bestState = None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def run_epoch(loader, training):
        model.train() if training else model.eval()

        totalLoss = 0.0
        correct = 0
        total = 0

        for xb, y_onehot in loader:
            xb = xb.to(device)              # (B, 3, 315)
            y_onehot = y_onehot.to(device)  # (B, 8)
            y = y_onehot.argmax(dim=1)      # (B,)

            xb = xb.transpose(1, 2)         # (B, 315, 3)

            if training:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                logits = model(xb)          # (B, 8)
                loss = criterion(logits, y)

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            totalLoss += loss.item() * xb.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += xb.size(0)

        return totalLoss / total, correct / total

    for epoch in range(1, epochs + 1):
        trainLoss, trainAcc = run_epoch(trainLoader, training=True)
        valLoss, valAcc = run_epoch(valLoader, training=False)

        if valAcc > bestValAccuracy:
            bestValAccuracy = valAcc
            bestState = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if log:
            print(f"Epoch {epoch:02d} | " f"train loss {trainLoss:.4f} acc {trainAcc:.3f} | " f"val loss {valLoss:.4f} acc {valAcc:.3f}")

    if bestState is not None:
        model.load_state_dict(bestState)

    bestTrainLoss, bestTrainAcc = run_epoch(trainLoader, training=False)
    training_performance = {"loss": float(bestTrainLoss), "accuracy": float(bestTrainAcc)}

    validation_performance = {
        "loss": float(valLoss),
        "accuracy": float(bestValAccuracy)
    }
    
    return model, training_performance, validation_performance


def u_wave_gesture_library_rnn_model(training_data_filepath):
    
    model, training_performance, validation_performance = train_rnn_model(training_data_filepath=training_data_filepath, num_epochs=50, num_hidden_states=128, num_layers=2, log=False)
    
    return model, training_performance, validation_performance


@torch.no_grad()
def test_cnn_model(model, dataset, batch_size=256, device=None, criterion=None, max_batches=None):
    """
    Tests a trained CNN on a dataset.

    Returns a dict with:
      - avg_loss
      - accuracy
      - num_samples
      - first_batch_shapes (input/output/target)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion is None:
        # Option B (one-hot labels) default
        criterion = torch.nn.BCEWithLogitsLoss()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    first_batch_shapes = None

    for b, (xb, y_onehot) in enumerate(loader):
        xb = xb.to(device)
        y_onehot = y_onehot.to(device)

        logits = model(xb)  # (B, 8)

        # Record shapes once
        if first_batch_shapes is None:
            first_batch_shapes = {
                "xb": tuple(xb.shape),
                "logits": tuple(logits.shape),
                "y": tuple(y_onehot.shape),
            }

        loss = criterion(logits, y_onehot)
        total_loss += loss.item() * xb.size(0)

        preds = logits.argmax(dim=1)
        true = y_onehot.argmax(dim=1)
        correct += (preds == true).sum().item()
        total += xb.size(0)

        if max_batches is not None and (b + 1) >= max_batches:
            break

    avg_loss = total_loss / total if total > 0 else float("nan")
    acc = correct / total if total > 0 else float("nan")

    return {
        "avg_loss": float(avg_loss),
        "accuracy": float(acc),
        "num_samples": int(total),
        "first_batch_shapes": first_batch_shapes,
    }


@torch.no_grad()
def test_rnn_model(model, dataset, batch_size=64, device=None, criterion=None, max_batches=None):
    """
    Tests a trained RNN on a dataset.

    Returns a dict with:
      - avg_loss
      - accuracy
      - num_samples
      - first_batch_shapes
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    totalLoss = 0.0
    correct = 0
    total = 0

    first_batch_shapes = None

    for b, (xb, y_onehot) in enumerate(loader):
        xb = xb.to(device)              # (B, 3, 315)
        y_onehot = y_onehot.to(device)  # (B, 8)
        y = y_onehot.argmax(dim=1)

        xb = xb.transpose(1, 2)

        logits = model(xb)              # (B, 8)

        if first_batch_shapes is None:
            first_batch_shapes = {
                "xb_after_transpose": tuple(xb.shape),
                "logits": tuple(logits.shape),
                "y": tuple(y.shape),
            }

        loss = criterion(logits, y)
        totalLoss += loss.item() * xb.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += xb.size(0)

        if max_batches is not None and (b + 1) >= max_batches:
            break

    avgLoss = totalLoss / total if total > 0 else float("nan")
    acc = correct / total if total > 0 else float("nan")

    return {
        "avg_loss": float(avgLoss),
        "accuracy": float(acc),
        "num_samples": int(total),
        "first_batch_shapes": first_batch_shapes,
    }

def rnn_experiments(training_data_filepath, test_data_file_path):

    num_hidden_states = [8, 16, 32, 64, 128, 256]
    train_results = []
    val_results = []
    test_results = []
    models = []
    
    best_num_hidden_states = -1
    best_test = -1
    
    print("\n--- Running Hidden States Experiments ---") 
    for n in num_hidden_states:
        
        # Training & validation
        model, train_result, val_result = train_rnn_model(
            training_data_filepath=training_data_filepath,
            num_epochs=50, 
            num_hidden_states=n,
            num_layers=2,
            log=False
        )
       
        # Test  
        ds = UWaveGestureLibraryDataset(test_data_file_path)
        results = test_rnn_model(model, ds, batch_size=32)
        
        # Store results  
        if results['accuracy'] > best_test:
            best_test = results['accuracy']
            best_num_hidden_states = n
            
        train_results.append(train_result['accuracy'])
        val_results.append(val_result['accuracy'])
        test_results.append(results['accuracy'])
        models.append(model)
    print(f"Best num hidden states: {best_num_hidden_states} | accuracy: {best_test}")
       
    # Plot Results 
    plt.figure(figsize=(8, 5)) 
    plt.plot(num_hidden_states, train_results, marker='o', linestyle='-', color='blue', label="Train")
    plt.plot(num_hidden_states, val_results, marker='o', linestyle='-', color='orange', label="Validation")
    plt.plot(num_hidden_states, test_results, marker='o', linestyle='-', color='Green', label="Test")
    plt.xlabel("Hidden States")
    plt.ylabel("Accuracy")
    plt.title("RNN Performance vs Hidden States")
    plt.legend()
    plt.show()
    # plt.savefig("rnn_hidden_states.png")
    # plt.clf() 
    
    
    # Number of epochs
    num_epochs = [10, 25, 50, 75, 100]
    train_results = []
    val_results = []
    test_results = []
    models = []
    best_test_epochs = -1
    best_num_epochs = -1
    
    print("\n--- Running Num Epochs Experiments ---") 
    for n in num_epochs:
        
        # Training & validation
        model, train_result, val_result = train_rnn_model(
            training_data_filepath=training_data_filepath,
            num_epochs=n, 
            num_hidden_states=128,
            num_layers=2,
            log=False
        )
       
        # Test  
        ds = UWaveGestureLibraryDataset(test_data_file_path)
        results = test_rnn_model(model, ds, batch_size=32)
        
        # Store results 
        if results['accuracy'] > best_test_epochs:
            best_test_epochs = results['accuracy']
            best_num_epochs = n
            
        train_results.append(train_result['accuracy'])
        val_results.append(val_result['accuracy'])
        test_results.append(results['accuracy'])
        models.append(model)
       
    print(f"Best num epochs: {best_num_epochs} | accuracy: {best_test_epochs}")
    # Plot Results 
    plt.figure(figsize=(8, 5)) 
    plt.plot(num_epochs, train_results, marker='o', linestyle='-', color='blue', label="Train")
    plt.plot(num_epochs, val_results, marker='o', linestyle='-', color='orange', label="Validation")
    plt.plot(num_epochs, test_results, marker='o', linestyle='-', color='Green', label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("RNN Performance vs Epochs")
    plt.legend()
    plt.show()
    # plt.savefig("rnn_epochs.png")
    # plt.clf()

if __name__ == "__main__":
   
    ''' NOTE: UNCOMMENT FOR LOCAL TESTING 
    train_path = "/home/andrew/cu/w26/comp4107/a3/UWaveGestureLibrary_TRAIN.csv"
    test_path = "/home/andrew/cu/w26/comp4107/a3/UWaveGestureLibrary_TEST.csv"

    print("\n--- CNN ---")
    model, train_perf, val_perf = u_wave_gesture_library_cnn_model(train_path)
    print("Train perf:", train_perf)
    print("Val perf:", val_perf)

    ds = UWaveGestureLibraryDataset(train_path)
    results = test_cnn_model(model, ds, batch_size=64)
    print("CNN Test results 1:", results)

    ds = UWaveGestureLibraryDataset(test_path)
    results = test_cnn_model(model, ds, batch_size=64)
    print("CNN Test results 2:", results)

    print("\n--- RNN ---")
    rnn_model, rnn_train_perf, rnn_val_perf = u_wave_gesture_library_rnn_model(train_path)
    print("RNN Training performance:", rnn_train_perf)
    print("RNN Validation performance:", rnn_val_perf)

    ds = UWaveGestureLibraryDataset(train_path)
    rnn_results = test_rnn_model(rnn_model, ds, batch_size=32)
    print("RNN TRAIN results:", rnn_results)

    ds = UWaveGestureLibraryDataset(test_path)
    results = test_rnn_model(rnn_model, ds, batch_size=32)
    print("RNN TEST results 2:", results)
    ''' 
    '''
    print("\n--- GRU ---")
    rnn_model, rnn_train_perf, rnn_val_perf = u_wave_gesture_library_rnn_model_gru(train_path)
    print("RNN w GRU Training performance:", rnn_train_perf)
    print("RNN w GRU Validation performance:", rnn_val_perf)

    ds = UWaveGestureLibraryDataset(train_path)
    rnn_results = test_rnn_model(rnn_model, ds, batch_size=32)
    print("RNN w GRU TRAIN results:", rnn_results)

    ds = UWaveGestureLibraryDataset(test_path)
    results = test_rnn_model(rnn_model, ds, batch_size=32)
    print("RNN w GRU TEST results 2:", results)
    '''
    ''' 
    print("\n--- Running Experiments ---") 
    rnn_experiments(training_data_filepath=train_path, test_data_file_path=test_path)
    '''
    