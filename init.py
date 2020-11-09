
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

'''
Can we predict the programming language given the Title and the body

Is there a corallation between title length, body length and creation date between rating?
Can we predict the quality rating of a stack over flow question?

creation_date_sanitized = np.array([int(datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timestamp()) for date in data['CreationDate']])
data['CreationDate'] = creation_date_sanitized
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
    
def predictingBodyLength(train_data):
    pass


