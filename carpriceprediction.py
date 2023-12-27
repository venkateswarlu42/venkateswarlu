#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("default")


# In[4]:


df = pd.read_csv("used_cars.csv")
df.head()


# In[5]:


df.to_csv('used_cars.csv', index = False)


# In[6]:


df.head()


# In[7]:


pd.set_option('display.max_columns', None)


# In[8]:


df.dtypes


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


df.drop('model', axis = 1, inplace = True)


# In[12]:


df.head()


# In[13]:


df.info()


# In[14]:


df['milage'] = df['milage'].str.replace(',', '').str.replace(' mi', '').astype(float)
df['milage']


# In[15]:


df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
df['price']


# In[16]:


df.head()


# In[17]:


most_frequent_value = df['fuel_type'].value_counts().idxmax()
most_frequent_value


# In[18]:


df['fuel_type'].fillna(most_frequent_value, inplace = True)


# In[19]:


df['fuel_type'].value_counts()


# In[20]:


df['fuel_type'].replace('–', 'Electric', inplace = True)


# In[21]:


df['fuel_type'].replace('not supported', 'Electric', inplace = True)


# In[22]:


df['fuel_type'].value_counts()


# In[23]:


df.dropna(subset = ['clean_title', 'accident'], axis = 0, inplace = True)


# In[24]:


df.shape


# In[25]:


import re
# Define a function to extract engine attributes
def extract_engine_attributes(engine_str):
    horsepower = re.search(r'(\d+\.\d+)HP|\d+\.\d+', engine_str)
    displacement = re.search(r'(\d+\.\d+L|\d+\.\d+ Liter)', engine_str)
    return horsepower.group(1) if horsepower else '',\
           displacement.group(1) if displacement else ''

# Apply the function to create new columns
df[['Horsepower', 'Engine_Displacement']] = df['engine'].apply(extract_engine_attributes).apply(pd.Series)


# In[26]:


df.head()


# In[27]:


df['Horsepower'].isnull().sum()


# In[28]:


df['Horsepower'] = pd.to_numeric(df['Horsepower'], errors = 'coerce')
df['Horsepower'].dtype


# In[29]:


df['Horsepower'].fillna(df['Horsepower'].mean(), inplace = True)
df['Horsepower'].isnull().sum()


# In[30]:


df['Engine_Displacement'] = df['Engine_Displacement'].str.replace('L', '')


# In[31]:


df['Engine_Displacement'] = pd.to_numeric(df['Engine_Displacement'], errors = 'coerce')


# In[32]:


df['Engine_Displacement'].fillna(df['Engine_Displacement'].mean(), inplace = True)
df['Engine_Displacement'].isnull().sum()


# In[33]:


df.head()


# In[34]:


df.drop('engine', axis = 1, inplace = True)


# In[35]:


df.shape


# In[36]:


df['age'] = 2023 - df['model_year']


# In[37]:


df.head()


# In[38]:


df.drop('model_year', axis = 1, inplace = True)


# In[39]:


df.head()


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a histogram
plt.figure(figsize=(8, 6))
plt.hist(df['price'], bins=20, color='skyblue', edgecolor='black')
plt.title('Price Distribution (Histogram)')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[41]:


plt.figure(figsize=(8, 6))
sns.kdeplot(df['price'], color='purple', shade=True)
plt.title('Price Distribution (Density Plot)')
plt.xlabel('Price')
plt.ylabel('Density')
plt.show()


# In[42]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a strip plot for the 'price' variable
plt.figure(figsize=(8, 6))
sns.stripplot(data=df, y='price', jitter=True, color='purple', alpha=0.5)
plt.title('Strip Plot: Price Distribution')
plt.ylabel('Price')
plt.grid(True)
plt.show()


# In[43]:


df['price'].describe()


# In[44]:


Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[45]:


df = df[~((df['price'] < (Q1 - 1.5 * IQR)) |(df['price'] > (Q3 + 0.7 * IQR)))]
df.shape


# In[46]:


#strip plot after removing outliers
import matplotlib.pyplot as plt
import seaborn as sns

# Create a strip plot for the 'price' variable
plt.figure(figsize=(8, 6))
sns.stripplot(data=df, y='price', jitter=True, color='purple', alpha=0.5)
plt.title('Strip Plot: Price Distribution')
plt.ylabel('Price')
plt.grid(True)
plt.show()


# In[47]:


#visualize fuel_type column in regrads to price
plt.figure(figsize = (10, 6))
plt.scatter(df['fuel_type'], df['price'])
plt.title('Scatterplot of Fuel Type vs Price')
plt.xlabel('Fuel Type')
plt.ylabel('Price')
plt.show()


# In[48]:


#create a boxplot of fuel_type vs price
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.boxenplot(x = 'fuel_type', y = 'price', data =df)
plt.xticks(rotation=45)


# In[49]:


df.shape


# In[50]:


import pandas as pd

# Calculate quartiles
Q1 = df['price'].quantile(0.25)
Q2 = df['price'].median()
Q3 = df['price'].quantile(0.75)

# Define a function to categorize data points into quartiles
def categorize_quartile(value):
    if value < Q1:
        return "Q1 (25%)"
    elif value < Q2:
        return "Q2 (50%)"
    elif value < Q3:
        return "Q3 (75%)"
    else:
        return "Q4 (100%)"

# Apply the categorize_quartile function to the 'price' column
df['Quartile'] = df['price'].apply(categorize_quartile)

# Count the number of data points in each quartile
quartile_counts = df['Quartile'].value_counts().reset_index()
quartile_counts.columns = ['Quartile', 'Count']

# Display the quartile distribution table
print(quartile_counts)


# In[51]:


df.describe()


# In[52]:


df['clean_title'].value_counts()


# In[53]:


# scatterplot of clean_title vs price
plt.figure(figsize = (10, 6))
plt.scatter(df['accident'], df['price'])
plt.title('Scatterplot of accident vs Price')
plt.xlabel('Accident Title')
plt.ylabel('Price')
plt.show()


# In[54]:


df.head()


# In[55]:


df['clean_title'].nunique()


# In[56]:


#drop clean_title column , accident column and Quartile column
df.drop(['clean_title', 'accident', 'Quartile'], axis = 1, inplace = True)


# In[57]:


df


# In[58]:


print(df['int_col'].nunique())
print(df['ext_col'].nunique())


# In[59]:


#create a group of transmission, fuel_type and price
df_group = df.groupby(['transmission', 'fuel_type'])['price'].mean().reset_index()
#create a pivot table
df_pivot = df_group.pivot(index = 'transmission', columns = 'fuel_type', values = 'price')
df_pivot


# In[60]:


#fill the null values with mean
df_pivot.fillna(df_pivot.mean(), inplace = True)
df_pivot


# In[61]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create a heatmap
fig, ax = plt.subplots(figsize=(18, 30), dpi=100)
cax = ax.matshow(df_pivot, cmap='RdBu')

# Label names
row_labels = df_pivot.columns
col_labels = df_pivot.index

# Move ticks and labels to the center
ax.set_xticks(np.arange(df_pivot.shape[1]))
ax.set_yticks(np.arange(df_pivot.shape[0]))

# Insert labels
ax.set_xticklabels(row_labels)
ax.set_yticklabels(col_labels)

# Rotate label if too long
plt.xticks(rotation=90)

# Add price numbers on each box
for i in range(len(col_labels)):
    for j in range(len(row_labels)):
        text = ax.text(j, i, f'{df_pivot.iloc[i, j]:.2f}', ha='center', va='center', color='w')

plt.colorbar(cax, label='Average Price')
plt.show()


# In[62]:


#draw and regression plot of milage vs price
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.regplot(x = 'milage', y = 'price', data =df)
plt.xticks(rotation=45)


# In[63]:


#draw and regression plot of age vs price
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.regplot(x = 'age', y = 'price', data =df)
plt.xticks(rotation=45)


# In[64]:


#draw a pairplot of all the features in df except brand, transmission
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.pairplot(df, hue = 'fuel_type')
plt.xticks(rotation=45)


# In[65]:


#draw a regressionplot of Horsepower vs price
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.regplot(x = 'Horsepower', y = 'price', data =df)
plt.xticks(rotation=45)


# In[66]:


#draw a scatter plot of Engine_Displacement vs price
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.regplot(x = 'Engine_Displacement', y = 'price', data =df)
plt.xticks(rotation=45)


# In[67]:


df.describe()


# In[68]:


#create a histogram for milage column
plt.figure(figsize=(8, 6))
plt.hist(df['milage'], bins=20, color='skyblue', edgecolor='black')
plt.title('Milage Distribution (Histogram)')
plt.xlabel('Milage')
plt.ylabel('Frequency')
plt.show()


# In[69]:


from scipy import stats
#to find pearson coefficent - Horsepower and Price
pearson_coef, p_value = stats.pearsonr(df['Horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[70]:


grp_test = df[['fuel_type','transmission','price']]
new_group_1 = grp_test[['transmission','price']].groupby(['transmission'])
new_group_1.head(5)


# In[71]:


#use f_oneway function to obtain anova values as:
f_val, p_val = stats.f_oneway(
    new_group_1.get_group('6-Speed A/T')['price'],
    new_group_1.get_group('8-Speed Automatic')['price'],
    new_group_1.get_group('7-Speed A/T')['price'],
    new_group_1.get_group('8-Speed A/T')['price'],
    new_group_1.get_group('9-Speed Automatic')['price'],
    new_group_1.get_group('10-Speed A/T')['price'],
    new_group_1.get_group('9-Speed A/T')['price'],
    new_group_1.get_group('Automatic CVT')['price'],
    new_group_1.get_group('7-Speed Automatic with Auto-Shift')['price'],
    new_group_1.get_group('10-Speed Automatic')['price'],
    new_group_1.get_group('6-Speed Automatic')['price'],
    new_group_1.get_group('8-Speed Automatic with Auto-Shift')['price'],
    new_group_1.get_group('7-Speed Automatic')['price'])
print( "ANOVA results: F=", f_val, ", P =", p_val)


# In[72]:


new_group_2 = grp_test[['fuel_type','price']].groupby(['fuel_type'])
new_group_2.head(5)


# In[73]:


f_val, p_val = stats.f_oneway(new_group_2.get_group('E85 Flex Fuel')['price'], new_group_2.get_group('Gasoline')['price'], new_group_2.get_group('Hybrid')['price'], new_group_2.get_group('Diesel')['price'], new_group_2.get_group('Plug-In Hybrid')['price'], new_group_2.get_group('Electric')['price']) 
   
print( "ANOVA results: F=", f_val, ", P =", p_val)


# In[74]:


df['fuel_type'].value_counts()


# In[75]:


#drop ext_col and int_col
df.drop(['ext_col', 'int_col'], axis = 1, inplace = True)
df.head()


# In[76]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
x = df[['brand']]
y = df[['price']]



# In[80]:


model = LinearRegression()


# In[81]:


grp_test = df[['fuel_type','transmission','price']]
new_group_1 = grp_test[['transmission','price']].groupby(['transmission'])
new_group_1.head(5)


# In[82]:


new_group_2 = grp_test[['fuel_type','price']].groupby(['fuel_type'])
new_group_2.head(5)


# In[87]:


#distribution plot of y_test and yhat3
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(yhat3, hist=False, color="b", label="Fitted Values")
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()


# In[88]:


#Perform polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[89]:


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


# In[90]:


pr = PolynomialFeatures(degree=3)
pr


# In[92]:


#create a linear regression model object for polynomial regression
poly = LinearRegression()
poly


# In[94]:


#import standard scaler and pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#create a list of tuples - each containing a model/estimator and its constructor
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]


# In[95]:


#pipe line object
pipe=Pipeline(Input)
pipe


# In[104]:


#distribution plot of y and ypipe
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(y,hist=False, color="b", label="Fitted Values")
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()


# In[111]:


#Distribution plot of y_test and ypipe2
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.distplot(y, hist=False, color="b", label="Actual Value")
sns.distplot(y, hist=False, color="r", label="Fitted Values")
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()


# In[118]:


#Lets visualize the model performance
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#First go for distribution plot on Training data
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
(y, "Actual Values (Train)", "Predicted Values (Train)", Title)


# In[ ]:





# In[ ]:




