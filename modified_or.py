# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:53:39 2020

@author: brend
"""

import os
import time
import itertools
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


trainData = pd.read_csv('D:\\INFR3700-FinalProject\\testing\\train.csv')
validData = pd.read_csv('D:\\INFR3700-FinalProject\\testing\\valid.csv')

frames = [trainData, validData]
data = pd.concat(frames)

data = data.drop(['Id', 'Tags', 'CreationDate'], axis=1)
data['Y'] = data['Y'].map({'LQ_CLOSE':0, 'LQ_EDIT': 1, 'HQ':2})
data.head()

labels = ['HQ', 'LQ_CLOSE', 'LQ_EDIT']
plt.style.use('classic')
plt.figure(figsize=(8, 8))
plt.pie(x=[len(data[data['Y'] == 2]), len(data[data['Y'] == 0]), len(data[data['Y'] == 1])], labels=labels, autopct="%1.3f%%")
plt.title("Question Distribution")
plt.show()

data['text'] = data['Title'] + ' ' + data['Body']
data = data.drop(['Title', 'Body'], axis=1)
data.head()

import re
from sklearn.preprocessing import label_binarize

def sanitize_text(text):
    text = text.lower()
    text = re.sub(r'[^(a-zA-Z)\s]','', text)
    return text
data['text'] = data['text'].apply(sanitize_text)

Y = label_binarize(data['Y'], classes=[0, 1, 2])
X = data['text']
n_classes = Y.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5, random_state=0)

vectorizer = TfidfVectorizer()
trainX = vectorizer.fit_transform(X_train)
validX = vectorizer.transform(X_test)

classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, penalty='l1'), n_jobs=-1)
y_score = classifier.fit(trainX, validX).decision_function(validX)

precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])