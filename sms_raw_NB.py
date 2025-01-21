# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:40:47 2024

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

email_data=pd.read_csv("E:/Data Science/13-Naive Bayes/sms_raw_NB.csv",encoding="ISO-8859-1")

#These are in text form ,open data frame and there are ham or spam
#cleaning the data
#The function tokenize the text and removes words
#with fewer than 4 characters.

import re

def cleaning_text(i):
    # Every thing else A-Z, a-z, or a space with a space
    i = re.sub("[^A-Za-z]+", " ", i).lower()
    
    w = []
    # Tokenize the string and filter out words shorter than 4 characters
    for word in i.split():
        if len(word) > 3:
            w.append(word)
    
    # Join the words back into a single string with spaces
    return " ".join(w)

#Testing above function with sample text
    cleaning_text("Hope you are having good week.just checking")
    cleaning_text('Hope I can understand your feelings 123123123 hi how are you')
    cleaning_text('Hi how are you ,I am sad')


#note the dataframe size is 5559 ,2 now removing empty spaces
#removing empty rows

email_data.text=email_data.text.apply(cleaning_text)
email_data=email_data.loc[email_data.text !='',:]  
email_data.shape

#you can use count vectorizer which directly converts a collection of 

#first we will split the data
from sklearn.model_selection import train_test_split
email_train,email_test=train_test_split(email_data, test_size=0.2)

#Splits each email into a list of words.
#Creating matrix of token count for entire dataframe

def split_into_words(i):
    return[word for word in i.split(" ")] 

#defining the preparation of email text into word count matrix format
#CountVectorizer:Converts the emails into a matrix of token counts.
#.fit():learns the vocabulary from the text data.text data into token count matrix.

emails_bow=CountVectorizer(analyzer=split_into_words).fit(email_data.text)
#for training messages
all_emails_matrix=emails_bow.transform(email_data.text)
#for training messages
train_email_matrix=emails_bow.transform(email_train.text)
#for testing messages
test_emails_matrix=emails_bow.transform(email_test.text)
#.tranform():convert 

tfidf_transformer=TfidfTransformer().fit(all_emails_matrix)
train_tfidf=tfidf_transformer.transform(train_email_matrix)
train_tfidf.shape



test_tfidf=tfidf_transformer.transform(test_emails_matrix)

test_tfidf.shape

#############################################################

#Now apply to naive bayes

from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()
classifier_mb.fit(train_tfidf,email_train.type)
 
#email train typle :this is the column in thet training dataset
#email_test that contains the target label
#which specify whether each msg is spam or ham

#label of corresponding label   


#evaluation on test data
test_pred_m=classifier_mb.predict(test_tfidf)

accuracy_test_m=np.mean(test_pred_m==email_test.type)
accuracy_test_m


from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m,email_test.type)
pd.crosstab(test_pred_m, email_test.type)

#Training Data Accuracy
train_pred_m=classifier_mb.predict(train_tfidf)

accuracy_test_m=np.mean(train_pred_m==email_test.type)
accuracy_test_m

#Test data(with laplace smoothing )

classifier_mb_lab=MB(alpha=3)
classifier_mb_lab.fit(train_tfidf,email_train.type)

test_pred_lap=classifier_mb_lab.predict(test_tfidf)
accuracy_test_lap=np.mean(test_pred_lap==email_test.type)
accuracy_test_lap
pd.crosstab(test_pred_lap,email_test.type)

from sklearn.metrics import accuracy_score

accuracy_score(test_pred_lap, email_test.type)
pd.crosstab(test_pred_lap, email_test.type)


#training Data accuracy
train_pred_lap=class