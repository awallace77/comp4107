#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
data = numpy.loadtxt(r"C:\Devel\COMP4107\bikerental_dataset.csv", delimiter=",", skiprows=1)
print(data.shape)


# In[3]:


x = data[:, :-1]
y = data[:, -1]
print(x.shape)
print(y.shape)


# In[4]:


train_size = int(0.8 * x.shape[0])

x_train = x[:train_size, :]
y_train = y[:train_size]
x_test = x[train_size:, :]
y_test = y[train_size:]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[20]:


import torch
class BicycleNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.normalization = torch.nn.LayerNorm(10)
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(10, 1)
        self.activation2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.normalization(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)

        return x

bicycle_network = BicycleNetwork()
print(bicycle_network)


# In[21]:


pred_test = bicycle_network(torch.tensor(x_test, dtype=torch.float32))
print(pred_test)


# In[22]:


loss_function = torch.nn.MSELoss()
loss = loss_function(torch.squeeze(pred_test), torch.tensor(y_test, dtype=torch.float32))
print(torch.sqrt(loss))


# In[23]:


optimizer = torch.optim.SGD(bicycle_network.parameters(), lr=0.001)


# In[26]:


x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

for epoch in range(30):
    y_train_pred = bicycle_network(x_train)
    loss = loss_function(y_train_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(epoch)
    print(loss)


# In[27]:


pred_test = bicycle_network(torch.tensor(x_test, dtype=torch.float32))
print(pred_test)

loss_function = torch.nn.MSELoss()
loss = loss_function(torch.squeeze(pred_test), torch.tensor(y_test, dtype=torch.float32))
print(torch.sqrt(loss))


# In[ ]:




