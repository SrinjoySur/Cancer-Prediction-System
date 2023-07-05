#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# In[2]:


model = LinearRegression()


# In[3]:


dataset = pd.read_csv('Downloads\cancer patient data sets.csv')


# In[4]:


y=dataset['Gender']
x=dataset['Air Pollution']


# In[5]:


x.shape


# In[6]:


X=x.values.reshape(1000,1)


# In[7]:


model.fit(X,y)


# In[8]:


model.predict([[2]])


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


plt.scatter(X,y)


# In[12]:


y_pred = model.predict(X)


# In[13]:


y_pred


# In[14]:


from sklearn import metrics


# In[16]:


metrics.mean_absolute_error(y, y_pred)


# In[ ]:




