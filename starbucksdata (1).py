#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("starbucks.csv")


# In[3]:


df.head()


# In[4]:


pd.merge(df.isna().sum().reset_index(), 
         df.dtypes.reset_index(), on='index')


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.columns = df.columns.str.strip()


# In[8]:


df['Beverage_category'].value_counts()


# In[9]:


df['Beverage_category'] = df['Beverage_category'].astype('category')


# In[10]:


df['Beverage'].value_counts()


# In[11]:


df['Beverage'] = df['Beverage'].astype('category')


# In[12]:


df["Beverage_prep"].value_counts()


# In[13]:


df['Beverage_prep'] = df['Beverage_prep'].astype('category')


# In[14]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df['Calories'], ax=axes[0])
axes[0].set_title('Box - Calories')
#Hist
sns.histplot(data=df['Calories'], ax=axes[1], kde=True)
axes[1].set_title('Hist - Calories')

plt.tight_layout()
plt.show()


# In[15]:


Q1 = df['Calories'].quantile(0.25)
Q3 = df['Calories'].quantile(0.75)
IQR = Q3 - Q1

df[df['Calories'] > Q3 + 1.5 * IQR]


# In[16]:


non_numeric_values = pd.to_numeric(df["Total Fat (g)"], errors='coerce').isnull()
df[non_numeric_values]


# In[17]:


fat  = [valor.replace(' ', ',') if ' ' in valor else valor for valor in df["Total Fat (g)"]]
df["Total Fat (g)"] = [float(valor.replace(',', '.')) for valor in fat]


# In[18]:


c = "Total Fat (g)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[19]:


Q1 = df[c].quantile(0.25)
Q3 = df[c].quantile(0.75)
IQR = Q3 - Q1

df[df[c] > Q3 + 1.5 * IQR]


# In[20]:


c = "Trans Fat (g)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[21]:


Q1 = df[c].quantile(0.25)
Q3 = df[c].quantile(0.75)
IQR = Q3 - Q1

df[df[c] > Q3 + 1.5 * IQR]


# In[22]:


c = "Saturated Fat (g)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True, bins = 3)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[23]:


Q1 = df[c].quantile(0.25)
Q3 = df[c].quantile(0.75)
IQR = Q3 - Q1

df[df[c] > Q3 + 1.5 * IQR]


# In[24]:


c = "Sodium (mg)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True, bins = 8)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[25]:


Q1 = df[c].quantile(0.25)
Q3 = df[c].quantile(0.75)
IQR = Q3 - Q1

df[df[c] > Q3 + 1.5 * IQR]


# In[26]:


c = "Total Carbohydrates (g)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[27]:


Q1 = df[c].quantile(0.25)
Q3 = df[c].quantile(0.75)
IQR = Q3 - Q1

df[df[c] > Q3 + 1.5 * IQR]


# In[28]:


c = "Cholesterol (mg)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[29]:


c = "Dietary Fibre (g)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True, bins = 8)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[30]:


Q1 = df[c].quantile(0.25)
Q3 = df[c].quantile(0.75)
IQR = Q3 - Q1

df[df[c] > Q3 + 1.5 * IQR]


# In[31]:


c = "Sugars (g)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True, bins = 8)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[32]:


Q1 = df[c].quantile(0.25)
Q3 = df[c].quantile(0.75)
IQR = Q3 - Q1

df[df[c] > Q3 + 1.5 * IQR]


# In[33]:


c = "Protein (g)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True, bins = 8)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[34]:


df["Vitamin A (% DV)"] = [float(valor.replace('%', '')) / 100 for valor in df["Vitamin A (% DV)"]]


# In[35]:


c = "Vitamin A (% DV)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True, bins = 8)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[36]:


Q1 = df[c].quantile(0.25)
Q3 = df[c].quantile(0.75)
IQR = Q3 - Q1

df[df[c] > Q3 + 1.5 * IQR]


# In[37]:


df["Vitamin C (% DV)"] = [float(valor.replace('%', '')) / 100 for valor in df["Vitamin C (% DV)"]]


# In[38]:


c = "Vitamin C (% DV)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c], ax=axes[1], kde=True, bins = 8)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[39]:


Q1 = df[c].quantile(0.25)
Q3 = df[c].quantile(0.75)
IQR = Q3 - Q1

df[df[c] > Q3 + 1.5 * IQR]


# In[41]:


cal_na = df[df["Beverage"] == "Iced Brewed Coffee (With Milk & Classic Syrup)"]
cal_na


# In[42]:


cal_sn = cal_na.dropna()
cal_sn['Caffeine (mg)'] = pd.to_numeric(cal_sn['Caffeine (mg)'], errors='coerce')


# In[43]:


df_numericas = cal_sn.select_dtypes(include=['float64', 'int64'])
matriz_correlacion = df_numericas.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Mapa de calor de correlación para variables numéricas')
plt.show()


# In[45]:


X_test = cal_sn['Calories'].values.reshape(-1, 1)
y_test = cal_sn["Caffeine (mg)"].values.reshape(-1, 1)


# In[46]:


lr = LinearRegression()
lr.fit(X_test, y_test)


# In[47]:


predicciones = lr.predict(X_test)

mse = mean_squared_error(y_test, predicciones)
print("Error cuadrático medio (MSE):", mse)

r2 = r2_score(y_test, predicciones)
print("Coeficiente de determinación (R^2):", r2)


# In[48]:


plt.figure(figsize=(8, 6))

plt.scatter(X_test, y_test, color='blue', label='Valores reales')

plt.plot(X_test, predicciones, color='red', label='Valores predichos')

plt.xlabel('Calories')
plt.ylabel('Caffeine')
plt.title('Regresión lineal')

plt.legend()

plt.show()


# In[49]:


n = cal_na[cal_na["Caffeine (mg)"].isnull()]
lr.predict(n["Calories"].values.reshape(-1,1))


# In[50]:


df['Caffeine (mg)'] = df['Caffeine (mg)'].fillna("96")
df['Caffeine (mg)'] = pd.to_numeric(df['Caffeine (mg)'], errors='coerce')


# In[51]:


c = "Caffeine (mg)"
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#Box
sns.boxplot(data=df[c][df[c] != -1], ax=axes[0])
axes[0].set_title(f'Box - {c}')
#Hist
sns.histplot(data=df[c][df[c] != -1], ax=axes[1], kde=True, bins = 8)
axes[1].set_title(f'Hist - {c}')

plt.tight_layout()
plt.show()


# In[52]:


Q1 = df[c].quantile(0.25)
Q3 = df[c].quantile(0.75)
IQR = Q3 - Q1

df[df[c] > Q3 + 1.5 * IQR]


# In[53]:


df.to_csv('starbucks_c.csv', index=False)


# In[54]:


df


# In[56]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Beverage_category', data=df, palette='viridis')
plt.title('Starbucks Stores')
plt.xlabel('Beverage_category')
plt.ylabel('Caffeine(mg)')
plt.show()


# In[59]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Calories', data=df, palette='viridis')
plt.title('Starbucks Stores')
plt.xlabel('Caffeine(mg)')
plt.ylabel('Beverage_category')
plt.show()


# In[ ]:




