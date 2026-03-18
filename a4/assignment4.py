import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

'''
  COMP4107: Neural Networks
  Assignment 4
  Group 111: 
    Andrew Wallace - 101210291
    Amr Altaweel - 101276934
  Due: March 20th, 2026
'''

class Linnaeus5Dataset(torch.utils.data.Dataset):
    """
        PyTorch dataset for the Linnaeus 5 dataset
    """
  

    def __init__(self, dataset_directory):
        """
            dataset_directory is the full path to the directory containing the dataset
        """
        data_transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize([0.5] * 3, [0.5] * 3) # scale to [-1, 1] for all three channels
        ])

        self.data = datasets.ImageFolder(root=dataset_directory, transform=data_transform)
        self.classes = self.data.class_to_idx
    
    def __len__(self):
        # num_samples is the total number of samples in the dataset
        num_samples = len(self.data)
        return num_samples


    def __getitem__(self, index):
        # index is the index of the sample to be retrieved
    
        # x is one sample of data
        # y is the same sample of data
        x, y = self.data[index]
        y = x
        return x, y
    
class Encoder(torch.nn.Module):
    def __init__(self, channels, hidden_channels=32, encoding_size=32):
        super(Encoder, self).__init__()
        
        hc = hidden_channels
       
        self.net = torch.nn.Sequential(
            # Hidden layer 1
            torch.nn.Conv2d(channels, out_channels=hc, kernel_size=3, stride=2, padding=1), # 32 -> 16
            torch.nn.BatchNorm2d(hc),
            torch.nn.ReLU(),
        
            # Hidden layer 2 
            torch.nn.Conv2d(hc, out_channels=2*hc, kernel_size=3, stride=2, padding=1), # 16 -> 8
            torch.nn.BatchNorm2d(2*hc),
            torch.nn.ReLU(),
            
            #Hidden Layer 3
            torch.nn.Conv2d(2*hc, out_channels=encoding_size, kernel_size=3, stride=2, padding=1), # 8 -> 4
        ) 
        
    def forward(self, x):
        x = self.net(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, channels, hidden_channels=32, encoding_size=32):
        super(Decoder, self).__init__()
        
        hc = hidden_channels
        
        self.net = torch.nn.Sequential(
            # Hidden Layer 1 
            torch.nn.ConvTranspose2d(encoding_size, 2*hc, kernel_size=3, stride=2, padding=1, output_padding=1), # 4 -> 8
            torch.nn.ReLU(),
        
            # Hidden layer 2 
            torch.nn.ConvTranspose2d(2*hc, hc, kernel_size=3, stride=2, padding=1, output_padding=1), # 8 -> 16
            torch.nn.ReLU(),
            
            # Hidden layer 3
            torch.nn.ConvTranspose2d(hc, hc, kernel_size=3, stride=2, padding=1, output_padding=1), # 16 -> 32
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(hc, channels, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
            # use tanh if normalized to [-1, 1]
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
        
class AutoEncoder(torch.nn.Module):
    """
        Defines an AutoEncoder
    """
    
    def __init__(self, channel_in, hidden_channels=16, out_channels=32):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(channel_in, hidden_channels, out_channels)
        self.decoder= Decoder(channel_in, hidden_channels, out_channels)
        
    def forward(self, x):
        encoding = self.encoder(x)
        x_hat = self.decoder(encoding)
        return x_hat, encoding
    
# 
def linnaeus5_autoencoder(training_data_directory):
    """
        A function that creates a cnn autoencoder model for images from the Linnaeus 5 dataset
    """
    
    # Device to use 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define parameters
    batch_size = 64
    lr = 1e-4
    epochs = 50
    noise_scale = 0.3
   
    # Dataset: Split into 75% train, 25% validation 
    dataset = Linnaeus5Dataset(training_data_directory)
    train_size = int(0.75 * dataset.__len__())
    val_size = dataset.__len__() - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
   
    # Model 
    model = AutoEncoder(channel_in=3, out_channels=512).to(device)

    # Loss function and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Sanity Check
    dataiter = train_loader._get_iterator()
    train_images = dataiter._next_data()[0]
    _, encoding = model(train_images)
    print(f"Encoded shape: {encoding.shape}")
    
    # Training 
    for epoch in range(epochs):
        model.train()
        
        total_loss = 0
        total_samples = 0
        
        for _, data in enumerate(train_loader):
           
            # Get input 
            x = data[0].to(device)
            
            # Add optional noise to image
            # random_sample = (torch.bernoulli((1 - noise_scale) * torch.ones_like(x)) * 2) - 1
            # noisy_x = random_sample * x
            
            # forward pass
            x_hat, encoding = model(x)

            # Calculate loss
            loss = loss_func(x_hat, x)
            
            # Step
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_samples += batch_size
        
        avg_train_loss = total_loss / total_samples
        
        # Validate 
        total_val_loss = 0
        total_val_samples = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                x = data[0].to(device)
                x_hat, _ = model(x)
                
                loss = loss_func(x_hat, x)
                
                total_val_loss += loss.item() * x.size(0)
                total_val_samples += batch_size 
        
        avg_val_loss = total_val_loss / total_val_samples
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    # -- End of Training --
    
    training_performance = avg_train_loss
    validation_performance =  avg_val_loss
    
    # model is a trained cnn autoencoder model for this task
    # training_performance is the performance of the model on the training set
    # validation_performance is the performance of the model on the validation set
    return model, training_performance, validation_performance

def linnaeus5_autoencoder_test(model, test_data_directory):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
   
    # Load dataset 
    test_dataset = Linnaeus5Dataset(dataset_directory=test_data_directory)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    total_mse = 0
    num_batches = 0
    
    loss_func = torch.nn.MSELoss()

    with torch.no_grad():  # no gradient needed for evaluation
        for x, _ in test_loader:
            x = x.to(device)
            x_hat, _ = model(x)

        # compute MSE loss
        batch_loss = loss_func(x_hat, x)
        total_mse += batch_loss.item()
        num_batches += 1

    test_mse = total_mse / num_batches
    return test_mse 

if __name__ == "__main__":

    # NOTE: Update this to the location of training data
    train_directory = "/home/andrew/cu/w26/comp4107/a4/Linnaeus 5 32X32/train" 
    test_directory = "/home/andrew/cu/w26/comp4107/a4/Linnaeus 5 32X32/test" 
   
    # Dataset 
    train_dataset = Linnaeus5Dataset(dataset_directory=train_directory)
    sample1, sample2 = train_dataset.__getitem__(0)
    print(f"Dataset Length: {train_dataset.__len__()}")
    print(f"Sample image shape: {sample1.shape}") 
    
    # Data loader 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dataiter = train_loader._get_iterator()
    train_images = dataiter._next_data()[0]
    print(f"Sample image (from data loader): {train_images.shape}")
   
    # To display a test image
    '''
    x, _ = train_dataset[0]
    # x = (x + 1) / 2 # if normalized to [-1, 1], unnormalize below 
    x = x.permute(1, 2, 0) # convert (C,H,W) → (H,W,C)
    
    plt.imshow(x)
    plt.axis('off')
    plt.show()
    ''' 
    
    # Train model
    model, training_performance, validation_performance = linnaeus5_autoencoder(train_directory)
    
    print(f"RESULTS: ")
    print(f"MODEL: {model}")
    print(f"TRAIN PERFORMANCE (MSE): {training_performance}")
    print(f"VAL PERFORMANCE (MSE): {validation_performance}")

    # Test model
    test_performance = linnaeus5_autoencoder_test(model, test_directory)
    print(f"TEST PERFORMANCE (MSE): {test_performance}")
    