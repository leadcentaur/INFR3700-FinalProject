
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

'''
Is there a corallation between title length, body length and creation date between rating?
Can we predict the quality rating of a stack over flow question?

creation_date_sanitized = np.array([int(datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timestamp()) for date in data['CreationDate']])
data['CreationDate'] = creation_date_sanitized

'''
'''
#We want to be able to predict the tags for a post given the body and title
def getTop15MostCommonTags(data_obj):
    dataTags = data_obj['Tags']
    for i in dataTags:
        tagStr = str(i).split(">")
        for j in tagStr:
            j.strip("<")
    print(tagStr)
    
    try:
       data_obj  = data_obj.drop(['Y'], axis = 1)
       data_obj = data_obj.drop(['Creation_Date'], axis = 1)
    except Exception as e:
        print("One or more of the columns being dropped failed: {}".format(e))
        pass
    return dataTags

'''
    
#Preprocessing for post prediction
trainData = pd.read_csv('D:\\INFR3700-FinalProject\\testing\\train.csv')
validData = pd.read_csv('D:\\INFR3700-FinalProject\\testing\\valid.csv')


frames = [trainData, validData]
data = pd.concat(frames)

mostCommonTags = getTop15MostCommonTags(data)

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

#Select all 2016 questions. we want to know of the 2016 year how many questions has a rating of HQ?
years = [2019, 2020]
for year in years:
    select_date = data.loc[data['Creation_Year'] == year]
    plt.hist(select_date['Y'], color='r', label="{} number of ratings ".format(str(year)))
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    plt.xlim(0,2.1)
    plt.legend()
    plt.show()

corr_matrix = data.corr()['Y']
print(corr_matrix.abs().sort_values(ascending=False))
max_cor_str = corr_matrix.abs().sort_values(ascending=False).index[1]

train, test = train_test_split(data, test_size=0.5, random_state=42)

plt.scatter(data['Title'], data['Body'], s=2, alpha=0.6)
plt.xlabel("")
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


