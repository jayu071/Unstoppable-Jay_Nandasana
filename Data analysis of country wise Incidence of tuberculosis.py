#!/usr/bin/env python
# coding: utf-8

# Data analysis of country wise Incidence of tuberculosis (per 100,000 people) using Machine Learning.
# 
# Project Objectives 
# 1. Early Recognition / Early Detection of improvement treatment zone for TB.  
# 2. Predicting improvement rate of TB in various country 
# 3. Major TB affected Country 
# 4. World wide Yearly improvement in TB cases. 

# In[79]:


import pandas as pd
import numpy as np
import seaborn as sb
import pandas as pd
df = pd.read_excel (r'P:\Jay\AI Project\AI Project\Data_TB\tb2excel.xls') 

#(use "r" before the path string to address special character, such as '\'). Don't forget to put the file name at the end of the path + '.xls'

print (df)


# In[ ]:


#import excel file as a dataframe "df"

import tkinter as tk
from tkinter import filedialog
import pandas as pd

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 300, height = 300, bg = 'lightsteelblue')
canvas1.pack()

def getExcel ():
    global df
    
    import_file_path = filedialog.askopenfilename()
    df = pd.read_excel (import_file_path)
    print (df)
    
browseButton_Excel = tk.Button(text='Import Excel File', command=getExcel, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=browseButton_Excel)

root.mainloop()


# In[80]:


df.describe()


# In[81]:


df.head()


# In[82]:


print(df.dtypes)


# In[83]:


df.columns


# DATA NORMALIZATION and Concert Object to Floate value

# In[84]:


df = pd.DataFrame(df)
df['2000 [YR2000]'] = pd.to_numeric(df['2000 [YR2000]'], errors='coerce')
df['2009 [YR2009]'] = pd.to_numeric(df['2009 [YR2009]'], errors='coerce')
df['2010 [YR2010]'] = pd.to_numeric(df['2010 [YR2010]'], errors='coerce')
df['2011 [YR2011]'] = pd.to_numeric(df['2011 [YR2011]'], errors='coerce')
df['2012 [YR2012]'] = pd.to_numeric(df['2012 [YR2012]'], errors='coerce')
df['2013 [YR2013]'] = pd.to_numeric(df['2013 [YR2013]'], errors='coerce')
df['2014 [YR2014]'] = pd.to_numeric(df['2014 [YR2014]'], errors='coerce')
df['2015 [YR2015]'] = pd.to_numeric(df['2015 [YR2015]'], errors='coerce')
df['2016 [YR2016]'] = pd.to_numeric(df['2016 [YR2016]'], errors='coerce')
df['2017 [YR2017]'] = pd.to_numeric(df['2017 [YR2017]'], errors='coerce')

print (df)
print(df.dtypes)


# In[85]:


#Create a new function:
def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print ("Missing values per column:")
print (df.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column

#Applying per row:
print ("\nMissing values per row:")
print (df.apply(num_missing, axis=1).head()) #axis=1 defines that function is to be applied on each row


# In[86]:


#Find NAN value in each coloum and raw

dfobj = pd.DataFrame(df)
print("NAN value in each coloum:") 
print(dfobj.isnull().sum())

print("\n NAN value in each raw:") 
for i in range(len(dfobj.index)) :
    print("Nan in row ", i , " : " ,  dfobj.iloc[i].isnull().sum())


# In[87]:


# making new data frame with dropped NA values 
df2 =df.dropna() 
  
# comparing sizes of data frames 
print("Old data frame length:", len(df), "\nNew data frame length:",  
       len(df2), "\nNumber of rows with at least 1 NA value: ", 
       (len(df)-len(df2))) 


# In[91]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
a=df2.plot(x ='Country Name', y='2012 [YR2012]', kind = 'bar')
b=df2.plot(x ='Country Name', y='2017 [YR2017]', kind = 'bar')
plt.show()


# In[92]:


#year wise TB change rate.
df.sum(axis=0,skipna = True)


# In[95]:


df2.plot(x ='Country Name',kind = 'bar')


# In[96]:


df3=df2.groupby(['Country Name']).mean()
print(df3)


# DATA VISULIZATION And ANALYSIS

# In[97]:


# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix( df);


# In[13]:


import seaborn as sns
sns.heatmap(df2.corr())


# In[90]:


sb.relplot(x='Country Name',y='2000 [YR2000]',data=df2,hue='Country Code')


# In[98]:


df2.max()


# In[99]:


#identify of country of maximun TB cases found in specific year and rate of change of TB cases.

display(df2[df2['2009 [YR2009]'] == df2['2009 [YR2009]'].max() ])

display(df2[df2['2000 [YR2000]'] == df2['2000 [YR2000]'].max() ])


# In[100]:


#set Country name to index

jay= df2.set_index("Country Name").select_dtypes('number')
jay.head()


# In[101]:


#identify of country of maximun TB cases found in each year

idx =jay.idxmax().drop_duplicates()
idx


# In[102]:


#highlight most significant country and each year maximum tb cases.
jay_max =jay.loc[idx]
jay_max.style.highlight_max()


# In[110]:


year2009=display('YEAR 2009 top 5 TB affected City:',df2.nlargest(5,'2009 [YR2009]').plot(x="Country Name",kind='bar'))


# In[111]:


year2012=display('YEAR 2012 top 5 TB affected City:',df2.nlargest(5,'2012 [YR2012]'))
year2017=display('YEAR 2017 top 5 TB affected City:',df2.nlargest(5,'2017 [YR2017]'))


# In[112]:


#TB cases rate in india 
print("TB CASES IN INDIA AND IMPROVEMENT RATE")
plt.subplot(2,1,1)
jay.loc["India"].plot(kind = 'pie' ,autopct='%1.1f%%')

#using line chart and bar chart
plt.subplot(2,1,2)
jay.loc["India"].plot(kind = 'bar')

plt.subplot(2,1,2)
jay.loc["India"].plot(kind = 'line')


# In[113]:


boxplot = jay.boxplot(column=['2000 [YR2000]', '2009 [YR2009]', '2010 [YR2010]', '2011 [YR2011]',
       '2012 [YR2012]', '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]',
       '2016 [YR2016]', '2017 [YR2017]'])


# In[24]:


data = jay.loc[["India"],:]
data


# In[138]:


data.plot(style='*')


# In[28]:


data2=data.append({'2000 [YR2000]':2000,
"2009 [YR2009]":2009,
"2010 [YR2010]":2010,
"2011 [YR2011]":2011,
"2012 [YR2012]":2012,
"2013 [YR2013]":2013,
"2014 [YR2014]":2014,
"2015 [YR2015]":2015,
"2016 [YR2016]":2016,
"2017 [YR2017]":2017,},ignore_index=True)


# In[37]:


aaa=data2.T
aaa


# In[40]:


data2=aaa.rename(columns={0:'IndiaTBcase',1:'Year'})
data2


# In[126]:


data2.plot(x='Year',y='IndiaTBcase',style='X')


# In[127]:


X=pd.DataFrame(data2['Year'])
y=pd.DataFrame(data2['IndiaTBcase'])


# In[149]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=1)


# In[150]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[151]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print('regressor.intercept_',regressor.intercept_)
print('regressor.coef_',regressor.coef_)


# In[152]:


y_test


# In[153]:


y_pred=regressor.predict(x_test)
y_pred


# In[154]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))


# In[155]:


print(regressor.score(x_test, y_test))


# In[139]:


y_pred = regressor.predict(x_test) 
plt.scatter(x_test, y_test, color ='b') 
plt.plot(x_test, y_pred, color ='k') 
  


# In[ ]:




