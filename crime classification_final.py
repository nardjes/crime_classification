#!/usr/bin/env python
# coding: utf-8

# # import the necessary modules

# In[2]:


import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn import metrics
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import timeit
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import the data set

# In[2]:


# Read in the data with `read_csv()`
#change the mane of the data frame to Incident_data 
data = pd.read_csv('/Users/macbook/Desktop/More Balanced Dataset.csv')
data.head()


# In[3]:


data.reset_index(drop=True)


# In[4]:


data['Address']= data['Address'].str.contains('/').replace([True, False ], ["1","0"], inplace= False)
data.head()


# In[5]:


data.shape


# In[6]:


data.Category.value_counts()


# # Preprocessing Data  using scikit-learn: 
# ## FEATURES

# In[7]:


data = data.drop(['Descript', 'Resolution', 'IncidntNum', 'Location', 'PdId', 'Address'], axis = 1)


# In[8]:


data.head()


# # scikit-learn understand only numeric data
# 
# ##### convert  Time   to datetime format (numeric format)
# 
# 

# In[9]:


data['Time'] = pd.to_datetime(data.Time)
data.dtypes


# ##### Extract Hour from Time

# In[10]:


data['Hour'] = data['Time'].dt.hour
data.head()


# ##### convert   Date   to datetime format
# 
# 

# In[11]:


data['Date'] = pd.to_datetime(data.Date)
data.dtypes


# ##### Extract Year from Date

# In[12]:


data['Year'] = data['Date'].dt.year
data.head()


# ##### Extract Month from Date

# In[13]:


data['Month'] = data['Date'].dt.month
data.head()


# ##### Extract Day from Date

# In[14]:


data['Day'] = data['Date'].dt.day
data.head()


# #### Encode PdDistrict 
# 

# In[15]:


PdDistrict_encoder =preprocessing.LabelEncoder()
data['DistrictEncoded'] = PdDistrict_encoder.fit_transform(data['PdDistrict'])
data.head()


# #### Convert crime category (labels) to numbers , encode it

# In[16]:


crime_encoder = preprocessing.LabelEncoder()
data ['CrimeEncoded'] = crime_encoder.fit_transform(data.Category)
data.head()


# ## Build new array and create train data and train label
# #### create respones 'category'

# In[17]:


crime_label = data['CrimeEncoded']
crime_label.head()


# In[18]:


data=data.drop(['Category','Date','Time','PdDistrict', 'DayOfWeek'],axis=1)
data.head()


# In[20]:


feature_cols = ['X', 'Y', 'Hour','Year','Month' ,'Day','DistrictEncoded']
crime_data= data[feature_cols]


# In[21]:


crime_data.head()


# In[22]:


crime_train_data, crime_test_data, crime_train_labels, crime_test_labels =train_test_split(crime_data, crime_label, test_size=0.3)


# In[23]:


crime_train_data.shape


# In[24]:


crime_test_data.shape


# # *RANDOM FOREST*

# In[25]:


start_time = time.time()
clf_rf = RandomForestClassifier()
clf_rf.fit(crime_train_data, crime_train_labels)
y_pred_rf = clf_rf.predict(crime_test_data)
log_loss_rf = log_loss(crime_test_labels, y_pred_rf)
print( 'random forest log_loss:' , log_loss_rf)
print("--- Time taken is %s seconds ---" % (time.time() - start_time))


# # *Gaussian Naive Bayes*
# 
# 

# In[26]:


start_time = time.time()
clf_gnb = GaussianNB()
clf_gnb.fit(crime_train_data, crime_train_labels)
y_pred_gnb = clf_gnb.predict(crime_test_data)
log_loss_gnb = log_loss(crime_test_labels, y_pred_gnb)
print ("Gaussian Naive Bayes log_loss: ",log_loss_gnb)
print("--- Time taken is %s seconds ---" % (time.time() - start_time))


# # *Logistic Regression*
# 
# 

# In[27]:


start_time = time.time()
logreg = LogisticRegression()
clf_logreg.fit(crime_train_data, crime_train_labels)
y_pred_logreg = clf_logreg.predict(crime_test_data)
log_loss_logreg = log_loss(crime_test_labels, y_pred_logreg)
print ("logistic regression log_loss: ",log_loss_logreg)
print("--- Time taken is %s seconds ---" % (time.time() - start_time))


# # *Nearest neighbors*

# In[28]:


start_time = time.time()
clf_knn = KNeighborsClassifier()
clf_knn.fit(crime_train_data, crime_train_labels)
y_pred_knn = clf_knn.predict(crime_test_data)
log_loss_knn = log_loss(crime_test_labels, y_pred_knn)
print ("nearest neighbors log_loss: ",log_loss_knn)
print("--- Time taken is %s seconds ---" % (time.time() - start_time))


# # *Decision Tree*
# 
# 

# In[29]:


start_time = time.time()
DecisTr = DecisionTreeClassifier()
clf_DecisTr.fit(crime_train_data, crime_train_labels)
y_pred_DecisTr = clf_DecisTr.predict(crime_test_data)
log_loss_DecisTr = log_loss(crime_test_labels, y_pred_DecisTr)
print ("Decision Tree log_loss: ",log_loss_DecisTr)
print("--- Time taken is %s seconds ---" % (time.time() - start_time))


# # *Support Vector Machine*

# In[30]:


start_time = time.time()
clf_svm = LinearSVC()
clf_svm.fit(crime_train_data, crime_train_labels)
y_pred_svm = clf_svm.predict(crime_test_data)
log_loss_svm = log_loss(crime_test_labels, y_pred_svm)
print ('Linear SVM log_loss:', log_loss_svm)
print("--- Time taken is %s seconds ---" % (time.time() - start_time))


# # *Ploting the comparaision of Log Lss using Histogram* 

# In[36]:


#seed is a function of numpy, which will give the same output each time it is executed.
np.random.seed(100)
names = ('rf','gnb','logreg','knn','DecisTr','svm')

bins = np.arange(len(names))
data= [log_loss_rf,log_loss_gnb,log_loss_logreg,log_loss_knn,log_loss_DecisTr,log_loss_svm]


plt.bar(bins, data, align='center', facecolor='blue')
plt.ylabel('log_loss')
plt.xlabel('Machine Learning Algorithms')
plt.title(r'Histogram')

plt.grid(True)


plt.show()


# # *Distribution of Longitude and Latitude in San Francisco map*
# 

# In[5]:


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16,16))
plt.ylim(37.71, 37.82)
plt.xlim(-122.360,-122.540)
ax.scatter(data['X'],data['Y'], s=0.01, alpha=1)


# # *Hotspot different crimes densities*
# 

# In[6]:


def newPalettes():
    palList = []
    colors = [sns.xkcd_rgb["light red"], "red","crimson","orange", "yellow", sns.xkcd_rgb["bright green"],
              "green", sns.xkcd_rgb["forest green"],"cyan","teal","navy","fuchsia","purple"]
    for c in colors:
        palList.append(sns.light_palette(c, as_cmap=True))

    return palList

palettes = newPalettes()


# In[7]:


# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
mapdata = np.loadtxt("/Users/macbook/Desktop/sf_map_copyright_openstreetmap_contributors.txt")
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]


data = pd.read_csv('/Users/macbook/Desktop/Police_Department_Incident.csv')

#train = train[1:300000]

#Get rid of the bad lat/longs
data['Xok'] = data[data.X < -121].X
data['Yok'] = data[data.Y < 40].Y
data["Date"] = pd.to_datetime(data["Date"], errors='raise')
data = data.dropna()


# In[8]:


#Select all rows with the given category, then split them by year.

trainP = data[data.Category == 'PROSTITUTION']

trainList = []
for i in range(2003, 2016):
    trainList.append(trainP[trainP.Date.dt.year == i])


# In[22]:


For this list of categories graphs are created. The categories PORNOGRAPHY/OBSCENE MAT, RECOVERED VEHICLE 
pl.figure(figsize=(20, 20 * asp))
for index, trainL in enumerate(trainList):
    pal = palettes[index]
    ax = pl.hexbin(trainL.Xok, trainL.Yok, cmap=pal,
                  bins=5,
                  mincnt=1)
    ax = sns.kdeplot(trainL.Xok, trainL.Yok, clip=clipsize,
                        cmap=pal,
                        aspect=(1 / asp))
        
ax.imshow(mapdata, cmap=pl.get_cmap('gray'),
              extent=lon_lat_box,
              aspect=asp)
pl.draw()


# In[23]:


categoryList = ["WARRANTS", "ASSAULT", "RUNAWAY"]


# In[44]:


#Distribution of crimes for each month

trainM = data[data.Category == "DRUNKENNESS"]
monthlyList = []
for i in range(1,13):
    monthlyList.append(trainM[trainM.Date.dt.month == i])


# In[45]:


pl.figure(figsize=(20, 20 * asp))
for index, mon in enumerate(monthlyList):
    pal = palettes[index]
    ax = pl.hexbin(mon.Xok, mon.Yok, cmap=pal,
                  bins=5,
                  mincnt=1)
    ax = sns.kdeplot(mon.Xok, mon.Yok, clip=clipsize,
                     cmap=pal,
                     aspect=(1 / asp))

ax.imshow(mapdata, cmap=pl.get_cmap('gray'),
              extent=lon_lat_box,
              aspect=asp)
pl.draw()

