#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


customers = pd.read_csv('Ecommerce Customers.csv')


# ## EDA

# In[15]:


customers.info()


# In[16]:


customers.head()


# In[17]:


customers.describe()


# In[18]:


sns.jointplot(x = 'Time on Website' , y = 'Yearly Amount Spent', data = customers)


# In[19]:


sns.jointplot(x = 'Time on App' , y = 'Yearly Amount Spent', data = customers)


# In[20]:


sns.pairplot(customers)


# In[21]:


sns.lmplot(x = 'Length of Membership' , y = 'Yearly Amount Spent' , data = customers)


# ## Linear Regression

# In[22]:


customers.columns


# In[23]:


X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
Y = customers['Yearly Amount Spent']


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# In[25]:


lm = LinearRegression()


# In[26]:


lm.fit(X_train, Y_train)


# In[27]:


predictions = lm.predict(X_test)
print(predictions)


# In[28]:


plt.scatter(x = Y_test , y = predictions)
plt.xlabel('Y Test')
plt.ylabel('Y Predicted')


# ## Model Evaluation

# In[31]:


MAE = metrics.mean_absolute_error(Y_test, predictions)
MSE = metrics.mean_squared_error(Y_test, predictions)
RMSE = np.sqrt(MSE)
print(f'MAE = {MAE} \nMSE = {MSE} \nRMSE = {RMSE}')


# In[32]:


sns.histplot((Y_test - predictions) , bins = 50)


# In[33]:


cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])
cdf


# In[35]:


cdf.describe()


# In[36]:


cdf.info()


# In[ ]:




