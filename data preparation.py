#!/usr/bin/env python
# coding: utf-8

# # import necessary modules
# 
# 

# In[372]:


import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import time


# ## Importing the data set
# 
# 

# In[769]:


data = pd.read_csv('/Users/macbook/Desktop/MÉMOIRE/dataset/Police_Department_Incident.csv')
data.head()


# # Data Exploration : understanding the data  what looks like,
# ## Returns the shape of the data frame in the form of a tuple (no. of rows, no. of cols).
# 
# 

# In[3]:


data.shape


# In[9]:


data.columns


# ## using the dtypes() method to display the different datatypes available
# 
# 

# In[4]:


data.dtypes


# ## Return a Series containing counts of unique values.
# 
# 

# In[5]:


data.Category.value_counts()


# ## Print a concise summary of a DataFrame.
# 
# 

# In[7]:


data.info()


# ## force pandas to calculate the true memory usage
# 
# 

# In[8]:


data.info(memory_usage='deep')


# # *Data cleaning*
# 
# ##### prepare the data: searching for missing values and duplicated values:
# 
# 
# 
# 

# ###### 1.searching for missing values:
# 
# 

# In[770]:


data.isna().sum()


# ###### make clearly the missing values in PdDisrict
# 
# 

# In[584]:


data[data.PdDistrict.isna()]


# ###### dealing with the missing values : drop it or fill it (i ll fill it with "unknown"):
# 
# 

# In[771]:


#inplace=true to make changes in the underlying data
data.PdDistrict.fillna(value = "UNKNOWN" , inplace= True)


# ###### check the missing values
# 
# 

# In[13]:


data.isna().sum()
#there is no missing values


# ###### check for the duplicated values;  by default use all of the columns
# 
# 

# In[14]:


# count the duplicate rows
data.duplicated().sum()
#So all rows are deferents between each other


# ###### show the ouliers
# 

# In[15]:


plt.scatter(x=data['X'], y=data['Y'])
plt.show()


# ###### filter the longitude and latitude coordinates and drop the outliers(long=-120.5 and lat =90)
# ###### Geographic coordinates of San Francisco, California, USA
# 
# 
# 
# 

# In[772]:



data[(data.X > -122.3649) & (data.Y > 37.81998)]


# In[31]:


data.shape


# In[773]:


data.drop(data[(data.X > -122.3649) & (data.Y > 37.81998)].index , inplace=True)


# ###### number of outliers : 2215024- 2214879=145
# 
# 

# In[73]:


data.shape


# # Data Reduction
# 
# 

# ###### we dont need RECOVERED VEHICLE, SECONDARY CODES, NON-CRIMINAL because it is not a kinda crime that we wanna predict in futur
# 

# In[84]:


data[data.Category == 'RECOVERED VEHICLE'].shape


# In[419]:


data.shape


# In[774]:


data.drop( data[ data['Category'] == "RECOVERED VEHICLE"].index , inplace=True)


# In[416]:


data.shape


# In[775]:


data.drop( data[ data['Category'] == "SECONDARY CODES"].index , inplace=True)


# In[425]:


data.shape


# In[776]:


data.drop( data[ data['Category'] == "NON-CRIMINAL"].index , inplace=True)


# In[590]:


data.shape


# ## Combining Similar Categories 

# In[777]:


#TREA and TRESPASS have same Descript so we combine them
data[data['Category'] == "TREA"]


# In[430]:


data.Category.value_counts()


# In[778]:


data.replace("TREA" , "TRESPASS", inplace = True)


# In[450]:


data.Category.value_counts()


# In[454]:


data[data['Category'] == "DISORDERLY CONDUCT"]


# In[455]:


data[data['Category'] == "PORNOGRAPHY/OBSCENE MAT"]


# In[779]:


data.replace("DISORDERLY CONDUCT" , "PORNOGRAPHY/OBSCENE MAT", inplace= True)


# In[453]:


data.Category.value_counts()


# In[456]:


data[data.Category == 'BAD CHECKS']


# In[458]:


data[data.Category == 'EMBEZZLEMENT ']


# In[780]:


data.replace("BAD CHECKS" , "Fraud/Counterfeiting", inplace= True)


# In[781]:


data.replace("EMBEZZLEMENT" , "Fraud/Counterfeiting", inplace= True)


# In[468]:


data.Category.value_counts()


# In[782]:


data.replace("VANDALISM" , "ARSON", inplace = True)


# In[478]:


data.Category.value_counts()


# In[727]:


data.Descript.value_counts()


# In[783]:


idx = data[data['Category'] == 'SEX OFFENSES, NON FORCIBLE'].index


# In[598]:


idx


# In[784]:


data.loc[4910 , 'Category']= 'Sexual Offenses'


# In[600]:


data.loc[4910 , 'Category']


# In[567]:


data.Category.value_counts()


# In[601]:


data[data.Category == 'Sexual Offenses']


# In[785]:


data.replace('SEX OFFENSES, FORCIBLE' , 'Sexual Offenses', inplace= True)


# In[603]:


data.Category.value_counts()


# In[786]:


data.replace('SEX OFFENSES, NON FORCIBLE' , 'Sexual Offenses', inplace= True)


# In[605]:


data.Category.value_counts()


# In[606]:


idx = data[data['Category'] == 'SUSPICIOUS OCC'].index


# In[607]:


idx


# In[608]:


data.loc[63 , 'Category']= 'Suspicious Person/act'


# In[609]:


data.loc[63 , 'Category']


# In[787]:


data.replace("SUSPICIOUS OCC" , "Suspicious Person/act", inplace= True)


# In[611]:


data.Category.value_counts()


# In[612]:


idx = data[data['Category'] == 'WEAPON LAWS'].index 


# In[613]:


idx


# In[788]:


data.loc[27 , 'Category']


# In[789]:


data.loc[27 , 'Category']= 'Deadly Tool Possession'


# In[790]:


data.loc[27 , 'Category']


# In[791]:


data.replace("WEAPON LAWS" , "Deadly Tool Possession", inplace= True)


# In[618]:


data.Category.value_counts()


# In[620]:


data[data.Category == 'Deadly Tool Possession']


# In[792]:


data.replace("BURGLARY" , "Deadly Tool Possession", inplace= True)


# In[622]:


data.Category.value_counts()


# In[793]:


data.replace("FRAUD" , "FORGERY/COUNTERFEITING", inplace= True)


# In[624]:


#i kept these categories depending on their descriptions
#the primary difference between theft (or larceny) and robbery is that robbery involves force. for that we dont combine em
#RUNAWAY from Justice
data.Category.value_counts()


# In[794]:


data.replace("FORGERY/COUNTERFEITING" , "Fraud/Counterfeiting", inplace= True)


# In[795]:


data.replace("DRIVING UNDER THE INFLUENCE" , "Traffic Violation", inplace= True)


# In[627]:


data.Category.value_counts()


# In[796]:


data.replace("WARRANTS" , "WARRANTS ISSUED", inplace= True)


# In[629]:


data.Category.value_counts()


# In[631]:


#looking to the description column
#these family offenses leads to missing person so i'll combine 'em 
data[data.Category == 'FAMILY OFFENSES']


# In[632]:


#
data[data.Category == 'MISSING PERSON']


# In[797]:


data.replace("FAMILY OFFENSES" , "MISSING PERSON", inplace= True)


# In[634]:


data.Category.value_counts()


# In[635]:


data[data.Category == 'LARCENY/THEFT']


# # Export the dataset to a CSV file:
# 
# 

# In[637]:


data.to_csv('Clean_data_crime.csv' ,  index= False)


# In[643]:


#i wanna know if the crimes really decrease in 2018 as the figure shows
data[data.Date == '04/30/2018']


# In[644]:


data['Date'] = pd.to_datetime(data.Date)
data.dtypes


# In[645]:


data['Year'] = data['Date'].dt.year
data.head()


# In[648]:


#in 2018 the number of crimes decrease in gigantic way from 134754 in 2003 to 40779 in 2018 (93975 variation)
data.Year.value_counts()


# In[798]:


#Combined classes with less than 2000 samples into OTHER OFFENSES category.
data.replace("GAMBLING" , "OTHER OFFENSES", inplace = True)


# In[799]:


data.replace("EXTORTION" , "OTHER OFFENSES", inplace = True)


# In[800]:


data.replace("BRIBERY" , "OTHER OFFENSES", inplace = True)


# In[801]:


data.replace("SUICIDE" , "OTHER OFFENSES", inplace = True)


# In[757]:


data.Category.value_counts()


# In[802]:


#Created a new category called VIOLENT/PHYSICAL CRIME where physical harm or guns were involved.
data.replace("ARSON" , "VIOLENT/PHYSICAL CRIME", inplace = True)


# In[759]:


data.Category.value_counts()


# In[803]:


data.replace("WEAPON LAWS" , "VIOLENT/PHYSICAL CRIME", inplace = True)


# In[804]:


data.replace("VANDALISM" , "VIOLENT/PHYSICAL CRIME", inplace = True)


# In[805]:


data.replace("ROBBERY" , "VIOLENT/PHYSICAL CRIME", inplace = True)


# In[806]:


data.Category.value_counts()


# In[810]:


data.to_csv('More Balanced Dataset.csv' , index =False)

