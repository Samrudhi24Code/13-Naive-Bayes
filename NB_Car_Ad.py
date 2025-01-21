# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 08:36:05 2024

@author: Dell
"""

'''
Problem Statement:
    THis dataset contains information of users in social network.
    THis social network has serveral business clients which can post 
    ads on it.
    One of the clients has a cars company which has just
    lauched a Luxury SUV for ridiculous price.
    Build a Beroulli Naive Bayes model using this dataset and classify
    which of the users of the social network are going to purchase
    this luxury SUV.1 implies that there was a purchase ND 0 impliethere
    s wasnt a purchase.
'''

'''
1.Buisness Problem
1.1 What is the business objective?
1.1.1 THis will help you bring these audiences to your 
website who are interested in cars.

ANd ,there will be many of those who are planning to buy a car in the
near future.
.
1.1.2.Communicating with your target audience over social
media is always a great way to build a good market reputation.
Try responding to peoples's  automobile related queries on twitter and facebook
pages quickly to be their first choice when it comes to buying a car.


1.2 Are there any contraits?

Not having a clear marketing or social media strategy may result
in reduced benefits for your business

Additional resources may be needed to manage your online prence.

Social media is immediate and needs daily monitoring

If you dont actively manage your socail media presence,
you may not see any real benefits

There is a risk of unwanted or unappropriate behaviour on your site 
inclusing bullying and harassment.


Greater exposure online has the potential to attract risks.
 Risks can include negative feedback information, Leaks, or hacking


'''
#2.Work on each features of the dataset  to create a data dictionary as

#User ID:Integer type is not contributing
#Gender:Object type to be label encoding
#Age:Integer
#EstimateSalary:Integer
#Purchase:Interger

###################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

car=pd.read_csv("E:/Data Science/13-Naive Bayes/NB_Car_Ad.csv")

#Exploratory data analsis
car.columns
car.dtypes
car.describe()

#min age of employee is 18 years
#max age of employee is 60 years
#avg age is 37.65
#min salary=15000
#avg salary=69724
#max salary is 69742

car.isna().sum()
car.drop(["User ID"],axis=1,inplace=True)
car.dtypes
plt.hist(car.Age)
#Age is normally distributed
#age is normally distributed
plt.hist(car.EstimatedSalary)
#data is normallyy distributed but right skewe


#data Pre-processing
#3.1 Data cleaning,Feature 

car.dtypes

from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()
#this one is the model of label_emcoder which one is applied to 
# all the object sdata type
car['Gender']=label_encoder.fit_transform(car['Gender'])

#Now let us apply normalization function

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

car_norm=norm_func(car)
car_norm.describe()

#Now let us desinate train data amd Test data

from sklearn.model_selection import train_test_split
car_train,car_test = train_test_split(car_norm,test_size=0.2)

col_names1=list(car_train.columns)
train_X=car_train[col_names1[0:2]]
train_y=car_train[col_names1[3]]
col_names2=list(car_train.columns)
test_X=car_test[col_names2[0:2]]
test_y=car_test[col_names2[3]]

#########################################################
# Model Building

#Build the model on the scaled data (try multiple options).
#Build a Naive Bayes model.

#Like Multinomialne, this classifier is suitable for discrete data. The

#BernoulLINB is designed for binary/boolean features.

from sklearn.naive_bayes import BernoulliNB as BB
classifer_bb=BB()

classifer_bb.fit(train_X,train_y)
#Let us now evaluate on test data
test_pred_b=classifer_bb.predict(test_X)
#Accuracy of the predication
accuracy_test_b=np.mean(test_pred_b==test_y)
accuracy_test_b
