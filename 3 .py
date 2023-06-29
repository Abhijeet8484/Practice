#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets


# In[2]:


iris = datasets.load_iris()
iris


# In[3]:


df = pd.DataFrame(iris['data'])
df.head()


# In[4]:


df[4] = iris['target']
df.head()


# In[8]:


# Adding column names 
df.rename(columns = {0:'SepalLengthCm',1:'SepalWidthCm', 2:'PetalLengthCm', 3:'PetalWidthCm', 4:'Species'}, inplace = True)
df.head()


# In[9]:


df.describe()


# In[10]:


df.shape


# In[11]:


df.mean()


# In[13]:


df.median()


# In[14]:


df.Species.mode()


# In[15]:


df.groupby(['Species']).count()


# In[16]:


df.SepalLengthCm.std()


# In[17]:


df.SepalWidthCm.std()


# In[18]:


df.PetalLengthCm.std()


# In[19]:


df.PetalWidthCm.std()


# In[ ]:




