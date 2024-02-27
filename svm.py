# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:14:22 2024

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
forest=pd.read_csv("forestfires.csv")
forest.dtypes
###############################################
#EDA
forest.shape
plt.figure(figsize=(16, 10))
sns.countplot(data=forest, x='month')
plt.show()
#Aug and sep has highest value
sns.countplot(forest.day)
#Friday sunday and saturday has highest value

sns.distplot(forest.FFMC)
#data is normal and slight left skewed
sns.boxplot(forest.FFMC)
#There are several outliers
sns.distplot(forest.DC)
#data is normal and slight left skiwed
sns.boxplot(forest.DC)
#There are outliers
sns.distplot(forest.ISI)
#data is normal
sns.boxplot(forest.ISI)
#there are outliers
sns.distplot(forest.wind)
#data is normal and slight right skewed
sns.boxplot(forest.wind)

#There are outliers
sns.distplot(forest.rain)
#data is normal
sns.boxplot(forest.rain)
#their are outliers


#Now let us check the highest fire in km?
forest.sort_values(by="area",ascending=False).head(5)
highest_fire_area=forest.sort_values(by="area",ascending=True)

plt.figure(figdize=(8,6))
plt.title("Temperature vs area of fire")
plt.bar(highest_fire_area['temp'],
        highest_fire_area['area'])
plt.xlabel("Temperature")
plt.ylabel("Area per km-sq")
plt.show()
#once the fire starts,almost 1000+ sq area's
#tempersture goes beyond 25 and
#around 750km area is facing temp 30+
#now let us check the highest rain in the forest

highest_rain=forest.sort_values(by='rain',ascending=False)[['month','day','rain']].head(5)
highest_rain
#highest rain observed in the month of aug
#let us check highest and lowest temperature in month and day
highest_temp=forest.sort_values(by='temp',ascending=False)[['month','day','rain']].head(5)

lowest_temp=forest.sort_values(by='temp',ascending=True)[['month','day','rain']].head(5)

print("Highest Temperature",highest_temp)
#Highest temp observed in aug
print("Lowest_temp",lowest_temp)
#lowest temperature in the month of dec

forest.isna().sum()
##########################################
#sal1.dtypes

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
forest.month=labelencoder.fit_transform(forest.month)
forest.day=labelencoder.fit_transform(forest.day)
forest.size_category=labelencoder.fit_transform(forest.size)

forest.dtypes
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["month"])
df_t=winsor.fit_transform(forest[["month"]])
sns.boxplot(df_t.month)


#######################################################
tc=forest.corr()
tc
fig,ax=plt.subplots()
fig.set_size_inches(200,10)
fig.heatmap(tc,annot=True ,cmap='YlGnBu')

#all the variables are moderately correlated with size_category

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test=train_test_split(forest,test_size=0.3)
train_X=train.iloc[:,:30]
train_y=train.iloc[:,:30]
test_X=test.iloc[:,:30]
test_y=test.iloc[:,:30]
#kernel linear
model_linear=SVC(kernel="linear")
model_linear.fit(train_X, train_y)
pred_test_linear=model_linear.predict(test_X)
np.mean(pred_test_linear==test_y)
#RBF
model_rbf=SVC(kernel='rbf')
model_rbf.fit(train_X,train_y)
pred_test_rbf=model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y)

from sklearn.model_selection import train_test_split
x=forest.drop(['FFMC','DMC'],axis='columns')
y=forest
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
len(x_test)
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)

model_C=SVC(gamma=1)
model_C.fit(x_train,y_train)
model_C.score(x_test,y_test)
