
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


'''
Can we predict the programming language given the Title and the body

Is there a corallation between title length, body length and creation date between rating?
Can we predict the quality rating of a stack over flow question?

creation_date_sanitized = np.array([int(datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timestamp()) for date in data['CreationDate']])
data['CreationDate'] = creation_date_sanitized
'''
'''
#We want to be able to predict the tags for a post given the body and title

#these are going to be the classifcaitions
def tagCounts(data_obj):
    dataTags = data_obj['Tags']
    tagList = []
    
    for i in dataTags:
        tagStr = str(i).replace(">",'').replace("<", ' ').split(" ")
        for tagName in tagStr:
            if tagName != '':
                tagList.append(tagName)
                
    tag_counter = {}
    for tag in tagList:
        if tag in tag_counter:
            tag_counter[tag] +=1
        else:
            tag_counter[tag] = 1
            
    values = []
    for key, value in tag_counter.items():
        values.append(value)
        
    most_common_tags = sorted(tag_counter, key=tag_counter.get, reverse=True)
    top15 = most_common_tags[:15]
    
    super_list = []
    tc_15 = values[:15]
    tn_15 = most_common_tags[:15]

    return sorted(tc_15, reverse=True), tn_15


def getTop15MostCommonTags(data_obj):
    dataTags = data_obj['Tags']
    tagList = []
    
    for i in dataTags:
        tagStr = str(i).replace(">",'').replace("<", ' ').split(" ")
        for tagName in tagStr:
            if tagName != '':
                tagList.append(tagName)
                
    tag_counter = {}
    for tag in tagList:
        if tag in tag_counter:
            tag_counter[tag] +=1
        else:
            tag_counter[tag] = 1
            
    most_common_tags = sorted(tag_counter, key=tag_counter.get, reverse=True)
    return most_common_tags[:15]


    
#Preprocessing for post prediction
trainData = pd.read_csv('D:\\INFR3700-FinalProject\\testing\\train.csv')
validData = pd.read_csv('D:\\INFR3700-FinalProject\\testing\\valid.csv')

frames = [trainData, validData]
data = pd.concat(frames)

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))

mostCommonTags = getTop15MostCommonTags(data)
tc = tagCounts(data)
test_values = tc[0]
test_keys = tc[1]

res = {} 
for key in test_keys: 
    for value in test_values: 
        res[key] = value 
        test_values.remove(value) 
        break
    
plt.figure(figsize=(15,6))
plt.hist(list(res.keys()), weights=list(res.values()), rwidth=0.5)
plt.title("Tags by populairty")
plt.show()

data = data.drop(['Id'], axis=1)
data = data.drop(['Tags'], axis=1)
#Test comment

ls = []
titleStr = data['Title']
qChecklst = np.array([])
contains_question_mark = np.array([ ls.append(1) if '?' in i else ls.append(0) for i in titleStr])
data['Contains_Question'] = ls


title_data_sanitzed = np.array([len(i) - i.count(" ") for i in data['Title']])
data['Title'] = title_data_sanitzed
body_data_sanitized = np.array([len(i) - i.count(" ") for i in data['Body']])
data['Body'] = body_data_sanitized

creation_date_sanitized = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in data['CreationDate']])
data['CreationDate'] = creation_date_sanitized

data['Creation_Day'], data['Creation_Month'], data['Creation_Year'] = data['CreationDate'].dt.day, data['CreationDate'].dt.month, data['CreationDate'].dt.year
data['Creation_Hour'], data['Creation_Minute'] =  data['CreationDate'].dt.hour,  data['CreationDate'].dt.minute
data = data.drop(['CreationDate'], axis=1)


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
quality_cat = data["Y"]
quality_cat_encoded = encoder.fit_transform(quality_cat)
data["Y"] = quality_cat_encoded
print(encoder.classes_)
# 0 HQ
# 1 LQ_CLOSE
# 2 LQ_EDIT


def exploringTheData_Part1():
    #Select all 2016 questions. we want to know of the 2016 year how many questions has a rating of HQ?
    years = [2020, 2016]
    for year in years:
        select_date = data.loc[data['Creation_Year'] == year]
        plt.hist(select_date['Y'], color='r', label="{} number of ratings ".format(str(year)))
        plt.xticks(ticks=[0,1,2], labels=["HQ","LQ_CLOSE","LQ_EDIT"])
        plt.xlabel("Rating")
        plt.title("Rating Count {}".format(str(year)))
        plt.ylabel("Number of ratings")
        plt.savefig("D:\\INFR3600-Final-Project\\hist_rating_count_{}.png".format(str(year)))
        plt.xlim(0,2.1)
        plt.legend()
        #plt.savefig("D:\\INFR3600-Final-Project\\hist_rating_count_{}.png".format(year))
        plt.show()
        
        
    #Which programming languages/tags were most common in year x?
corr_matrix = data.corr()['Y']
print(corr_matrix.abs().sort_values(ascending=False))
max_cor_str = corr_matrix.abs().sort_values(ascending=False).index[1]

plt.scatter(data['Title'], data['Body'], s=2, alpha=0.6)
plt.xlabel("Title Length")
plt.ylabel("Body Length")
plt.title("Body Length vs Title Length")
plt.savefig("D:\\INFR3600-Final-Project\\blen_vs_tlen.png")
plt.show()

#get contains question and has HQ rating
cq_var = data[data['Contains_Question'] == 1].iloc[:,2]
#get contains question and has LQ rating
dcq_var= data[data['Contains_Question'] == 0].iloc[:,2]

#of the titles that contain question marks, how many of those are high quality and how many our low quality
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.title("Contains ? and Ratings")
plt.xticks(ticks=[0,1,2], labels=["HQ","LQ_CLOSE","LQ_EDIT"])
plt.hist(cq_var)

plt.subplot(2,2,2)
plt.title("Does not contain ? and Ratings")
plt.xticks(ticks=[0,1,2] ,labels=["HQ","LQ_CLOSE","LQ_EDIT"])
plt.hist(dcq_var)
plt.tight_layout()
plt.show()
'''

#---- DATA PROCESSING START ----

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
trainX, validX, trainY, validY = train_test_split(data['text'], data['Y'], test_size=.2, random_state=0)

vectorizer = TfidfVectorizer()
trainX = vectorizer.fit_transform(trainX)
validX = vectorizer.transform(validX)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

'''
# ------ Logistic Regression classifier ------

lr_classifier = LogisticRegression(C=1.)
y_score = lr_classifier.fit(trainX, trainY)
print(f"Validation Accuracy of Logsitic Regression Classifier is: {(lr_classifier.score(validX, validY))*100:.2f}%")

y = label_binarize(data['Y'], classes=[0, 1, 2])
n_classes = y.shape[1]
print(n_classes)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[i], y_score.coef_.T[i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#tag_predictions = sgd_clf.predict(validX)
print(f"Validation Accuracy of LR: {(lr_classifier.score(validX, validY))*100:.2f}%")

scores = cross_val_score(lr_classifier, trainX, trainY, cv=3, scoring="accuracy")
print('Score for each fold:', scores)
plt.plot(scores, marker='o', linestyle='--')
plt.title("Score per fold")
plt.show()

pred = cross_val_predict(lr_classifier, trainX, trainY, cv=3)
print(pred.shape)

print('F1 Score macro:', f1_score(trainY, pred, average='macro'))
print('F1 Score micro:', f1_score(trainY, pred, average='micro'))
print('F1 Score weighted:', f1_score(trainY, pred, average='weighted'))


plt.plot(trainY, pred)
plt.show()

print('Confusion matrix: [[TN, FP],[FN,TP]', confusion_matrix(trainY, pred))
print('Classes:', lr_classifier.classes_)
'''
# ------ Logistic Regression classifier ------


# ------ SGDClassifier ------
'''
sgd_clf = SGDClassifier(random_state=42, max_iter=30, tol=1e-3)
y_score = sgd_clf.fit(trainX, trainY)

y = label_binarize(data['Y'], classes=[0, 1, 2])
n_classes = y.shape[1]
print(n_classes)

# Compute ROC curve and ROC area for each class

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[i], y_score.coef_.T[i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

scores = cross_val_score(sgd_clf, trainX, trainY, cv=3, scoring="accuracy")
print('Score for each fold:', scores)
plt.plot(scores, marker='o', linestyle='--')
plt.title("Score per fold")
plt.show()


#tag_predictions = sgd_clf.predict(validX)

print(f"Validation Accuracy of SDG: {(sgd_clf.score(validX, validY))*100:.2f}%")

pred = cross_val_predict(sgd_clf, trainX, trainY, cv=3)

print('F1 Score macro:', f1_score(trainY, pred, average='macro'))
print('F1 Score micro:', f1_score(trainY, pred, average='micro'))
print('F1 Score weighted:', f1_score(trainY, pred, average='weighted'))

print('Confusion matrix: [[TN, FP],[FN,TP]', confusion_matrix(trainY, pred))
print('Classes:', sgd_clf.classes_)


#from sklearn.metrics import precision_recall_curve

#predictingBodyLength()
'''
# ------ OneVsRest ------

classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, penalty='l1'), n_jobs=-1)
y_score = classifier.fit(trainX, trainY).decision_function(validX)

from sklearn import preprocessing
y = preprocessing.label_binarize(trainY, classes=[0, 1, 2])

predictions = classifier.predict(trainX)
print(predictions)
print(f"Validation Accuracy of SDG: {(classifier.score(validX, validY))*100:.2f}%")

y_scores = cross_val_predict(classifier, trainX, y, cv=3, method="decision_function")

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y[i], y_score[i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
fpr["micro"], tpr["micro"], _ = roc_curve(y[0].ravel(), y_score[0].ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
plt.show()

# ------ OneVsRest ------



