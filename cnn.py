#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[5]:


raw_input = torch.randint(0, 4, (1, 1000, 1))
f_input = nn.functional.one_hot(raw_input.squeeze(-1).long(), num_classes=4)
f_input = f_input.permute(0, 2, 1).float()
print(f_input.shape)


# In[6]:


class SignalCNN(nn.Module):
    def __init__(self):
        super(SignalCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=25, stride=1, padding=12)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# In[ ]:





# In[8]:


model = SignalCNN()
output = model(f_input)
print(output.shape)


# In[ ]:





# In[ ]:





# In[ ]:




