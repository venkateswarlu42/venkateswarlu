#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pyforest


# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# In[4]:


data=pd.read_csv("Housing.csv")


# In[5]:


data


# In[6]:


data.head(5)


# In[7]:


data.tail(5)


# In[8]:


X = data.drop('stories', axis=1)
y = data['bedrooms']


# In[9]:


X


# In[10]:


data.info()


# In[11]:


data.isnull().sum()


# In[12]:


np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 2


# In[13]:


X,y


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[16]:


model=LinearRegression()


# In[17]:


model.fit(X_train_scaled, y_train)


# In[18]:


y_pred = model.predict(X_test_scaled)


# In[19]:


print("Is there any Duplicate Values-->", data.duplicated().any())


# In[20]:


mse = mean_squared_error(y_test, y_pred)


# In[21]:


mse


# In[22]:


print(f'Mean Squared Error: {mse}')


# In[23]:


X=data.iloc[:, :-1]
y=data["bedrooms"]


# In[24]:


y


# In[25]:


pip install tensorflow


# In[26]:


np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 2


# In[27]:


X,y


# In[28]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[29]:


X_train_scaled
X_test_scaled


# In[30]:


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])


# In[31]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[32]:


model.fit(X_train_scaled, y_train, epochs=99, batch_size=30, verbose=0)


# In[33]:


mse = model.evaluate(X_test_scaled, y_test)


# In[34]:


mse


# In[35]:


plt.scatter(X, y, color='b', label='data')

plt.plot(X, model.predict(X), color='r', label='Best-fit Line')


plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.show()


# In[44]:


plt.bar(data['area'], data['price'], color='blue')
plt.xlabel('area')
plt.ylabel('price')
plt.title('Housepredction ')
plt.show()


# In[46]:


plt.pie(data['area'], labels=data['price'], autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of houses ')
plt.show()


# In[47]:


data = {
    'Values': np.random.randn(1000)  
}


# In[48]:


data


# In[51]:


plt.hist(data['Values'], bins=20, color='blue', edgecolor='black')
plt.xlabel('price')
plt.ylabel('area')
plt.title('Housing data')
plt.show()


# In[ ]:




