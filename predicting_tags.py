# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:28:31 2020

@author: brend
"""


import os
import time
import itertools
import re
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


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

# ---------- CODE START ------------

trainData = pd.read_csv('D:\\INFR3700-FinalProject\\testing\\train.csv')
validData = pd.read_csv('D:\\INFR3700-FinalProject\\testing\\valid.csv')

frames = [trainData, validData]
data = pd.concat(frames)


#  ---------- TAG COUNTS FUNCTION ------------

dataTags = data['Tags']
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
print("The top 15 most common tags are: {}".format(top15))

#  ---------- TAG COUNTS FUNCTION ------------

#Display top most common tags using word cloud
from wordcloud import WordCloud

tags_object = pd.DataFrame()
tags_object['tag_freq'] = tag_counter.values()
tags_object['tag_name'] = tag_counter.keys()
sorted_tags_object = tags_object.sort_values(by=['tag_freq'], ascending=False)

wordcloud = WordCloud(background_color='black', width=1200, height=800, relative_scaling='auto', random_state=42)
word_cloud_image = wordcloud.generate_from_frequencies(tag_counter)

fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=2)
fig.savefig("tag.png")
plt.show()

i=np.arange(len(top15))

sorted_tags_object.head(len(top15)).plot(kind='bar', alpha=0.8, color='red', edgecolor='blue')
plt.title('Top 15 tags')
plt.xticks(i, sorted_tags_object['tag_name'])
plt.xlabel('Tags')
plt.ylabel('Frequency')
#plt.savefig("top15.png")
plt.show()
#tags = vec.get_fre_nam()eatues


#Here we strip the html tags from the body of the question
final_data = pd.DataFrame()
def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result

def remove_tags_from_tags(string):
    result = re.sub(r'[^\w]',' ',string)
    return result

def categorize(string):
    
    return result

final_data['title_question'] = data['Title']
final_data['body_nohtml']=data['Body'].apply(lambda cw : remove_tags(cw))
final_data['tags_nohtml']=data['Tags'].apply(lambda f : remove_tags_from_tags(f))

vec = CountVectorizer(tokenizer = lambda x: x.split())
tag_mtlb = vec.fit_transform(final_data['tags_nohtml'])

    
'''
train, test = train_test_split(final_data, test_size=0.2, shuffle=True)

train_labels_mc = train['Hobby'] #try but including the income and >100k
test_labels_mc = test['Hobby']



print(tag_dtm)
'''





