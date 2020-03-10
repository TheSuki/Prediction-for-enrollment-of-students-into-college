#!/usr/bin/env python
# coding: utf-8

# In[17]:


# import pandas and matplotlib 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np


# In[35]:


from matplotlib import pyplot as plt
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)


# In[18]:


df = pd.read_csv(r"C:\Study\6th Sem\Innovation lab\Admission Prediction for students.csv")
df.head()


# In[19]:


# creating a histogram for the numeric data 
df.hist() 
  
# showing the  plot 
plt.show() 


# In[19]:



# scatter plot between Twelft Score and  Chamce of getting admission in the college 
plt.scatter(df['Twelft Score'], df['Chance of Admission']) 
plt.show() 


# In[22]:


df.rename(columns ={'Chance of Admission':'Chance of Admission','LOR':'LOR'}, inplace=True)
df.drop(labels='Serial No.',axis=1, inplace=True)


# In[23]:


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap= 'Blues')


# In[28]:


#Comparing CGPA with the Chance of Admission to check which students vcan get into the college

plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.distplot(df['CGPA'])
plt.title('CGPA Distribution of Applicants')


plt.subplot(1,2,2)
sns.regplot(df['CGPA'], df['Chance of Admission'])
plt.title('CGPA vs Chance of Admission')


# In[31]:


targets = df['Chance of Admission']
features = df.drop(columns= {'Chance of Admission'})

x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)


# In[ ]:





# In[37]:


scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.fit_transform(x_test)


# In[45]:


linreg = LinearRegression()
linreg.fit(x_train, y_train)
y_predict = linreg.predict(x_test)

linreg_score = (linreg.score(x_test, y_test))*100
linreg_score

axis = sns.distplot(y_test, hist=False, color="b", label="Actual Value")
sns.distplot(y_predict,hist=False,color="r",label="Fitted Values", ax=axis)


# In[ ]:





# In[ ]:




