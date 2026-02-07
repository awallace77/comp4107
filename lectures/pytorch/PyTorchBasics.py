#!/usr/bin/env python
# coding: utf-8

# In[34]:


import torch


# In[35]:


a = torch.tensor(1.23)
print(a)
print(a.shape)


# In[36]:


b = torch.tensor([1, 2, 3, 4])
print(b)
print(b.shape)


# In[37]:


c = torch.tensor([[1, 2], [3, 4]])
print(c)
print(c.shape)


# In[38]:


d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(d)
print(d.shape)


# In[39]:


torch.reshape(d, [4, 2])


# In[40]:


print(torch.mul(c, c))
print(torch.matmul(c, c))


# In[41]:


# The sequential way
model = torch.nn.Sequential(
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 5),
    torch.nn.Sigmoid()
)
print(model)


# In[42]:


# The modular way
class MyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.activation1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(10, 5)
        self.activation2 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)

        return x

my_network_instance = MyNetwork()
y = my_network_instance(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32))
print(my_network_instance)
print(y)


# In[ ]:





# In[ ]:




