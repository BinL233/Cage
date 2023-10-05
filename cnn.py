#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# In[2]:


# Create random values
raw_input = np.random.randint(0, 4, size=(1000, 1))

# One-hot
encoder = OneHotEncoder(categories='auto', sparse_output=False)
f_input = encoder.fit_transform(raw_input)

# Convert np matrix to torch matrix
f_input = torch.tensor(f_input, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
print(f_input.shape)
print(f_input)


# In[3]:


class SignalCNN(nn.Module):
    def __init__(self):
        super(SignalCNN, self).__init__()
        
        # layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=25, stride=1, padding=12)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# In[ ]:





# In[5]:


model = SignalCNN()
output = model(f_input)
print(output.shape)
print(output)


# In[ ]:





# In[ ]:





# In[ ]:




