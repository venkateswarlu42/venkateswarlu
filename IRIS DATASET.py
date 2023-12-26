#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyforest


# In[2]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[3]:


data=pd.read_csv("IRIS.csv")


# In[4]:


data.head()


# In[5]:


data.tail()


# In[10]:


data.describe()


# In[11]:


data.info()


# In[12]:


data.isnull().sum()


# In[13]:


X = data.drop('species', axis=1)
y = data['sepal_length']


# In[16]:


X,y


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


scaler = StandardScaler()


# In[19]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[20]:


X_train


# In[21]:


X_test


# In[22]:


model = Sequential([
    Dense(8, activation='relu', input_shape=(4,)),
    Dense(3, activation='softmax')
])


# In[23]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[33]:


model=LinearRegression()



# In[38]:


model.fit(X_train, y_train)


# In[40]:


y_pred = model.predict(X_test)


# In[41]:


y_pred


# In[42]:


np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 2


# In[43]:


X,y


# In[44]:


print("Is there any Duplicate Values-->", data.duplicated().any())


# In[49]:


def Mean_Squared_Error(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean()
    return mse


# In[50]:


mse = Mean_Squared_Error(y_test, y_pred)


# In[51]:


mse


# In[54]:


X=data.iloc[:, :-1]
y=data["sepal_length"]


# In[55]:


y


# In[71]:


model = LinearRegression()
model.fit(X, y)
model = LinearRegression()
model.fit(X, y)
plt.scatter(X, y, color='b', label='data')
plt.plot(X, y, color='r', label='Best-fit Line')

plt.xlabel('X')
plt.show()


# In[73]:


plt.bar(data['sepal_width'], data['sepal_length'], color='blue')
plt.xlabel('sepal_width')
plt.ylabel('sepal_length')
plt.title('IRIS DATA')
plt.show()


# In[74]:


plt.bar(data['petal_width'], data['petal_length'], color='blue')
plt.xlabel('petal_width')
plt.ylabel('petal_length')
plt.title('IRIS DATA')
plt.show()


# In[82]:


data = {
    'Values': np.random.randn(1000)  
}


# In[83]:


data


# In[89]:


labels = ['sepal_length', 'sepal_width', 'petal_length']

# Plotting the pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green', 'blue'])

# Adding a title
plt.title('IRIS DATSET')

# Display the plot
plt.show()


# In[ ]:




