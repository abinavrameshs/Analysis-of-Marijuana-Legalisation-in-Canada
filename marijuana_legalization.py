#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:04:22 2019

@author: abinavrameshsundararaman
"""
import time
import datetime
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import itertools
from bs4 import BeautifulSoup
import networkx as nx
from nltk.corpus import stopwords
import nltk
from nltk.corpus import reuters
from nltk.corpus import wordnet
import collections
import os, csv, lda, nltk
import pandas as pd
import numpy as np
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import PunktSentenceTokenizer,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

pre_legalization_legalizationcanada=pd.read_csv("pre_legalization_legalizationcanada.csv")
pre_legalization_MarijuanaCanada=pd.read_csv("pre_legalization_MarijuanaCanada.csv")
pre_legalization=pd.read_csv("pre_legalization.csv")

pre_tweets=pd.concat([pre_legalization_legalizationcanada, pre_legalization_MarijuanaCanada,pre_legalization]).drop_duplicates()


pre_tweets.to_csv("pre_legalization_all.csv", sep=',', encoding='utf-8')


##
ns_1=pd.read_csv("ns_1.csv")
ns_2=pd.read_csv("ns_2.csv")
ns_3=pd.read_csv("ns_3.csv")
ns_4=pd.read_csv("ns_4.csv")
ns_5=pd.read_csv("ns_5.csv")
ns_6=pd.read_csv("ns_6.csv")


ns_all=pd.concat([ns_1,ns_2,ns_3,ns_4,ns_5,ns_6]).drop_duplicates()
ns_all['Province']="NS"
ns_all.to_csv("ns_all.csv", sep=',', encoding='utf-8')

##
alberta_1=pd.read_csv("alberta_1.csv")
alberta_2=pd.read_csv("alberta_2.csv")
alberta_3=pd.read_csv("alberta_3.csv")
alberta_4=pd.read_csv("alberta_4.csv")
alberta_5=pd.read_csv("alberta_5.csv")
alberta_6=pd.read_csv("alberta_6.csv")


alberta_all=pd.concat([alberta_1,alberta_2,alberta_3,alberta_4,alberta_5,alberta_6]).drop_duplicates()
alberta_all['Province']="AB"
alberta_all.to_csv("alberta_all.csv", sep=',', encoding='utf-8')

##
saskatchewan_1=pd.read_csv("saskatchewan_1.csv")
saskatchewan_2=pd.read_csv("saskatchewan_2.csv")
saskatchewan_3=pd.read_csv("saskatchewan_3.csv")
saskatchewan_4=pd.read_csv("saskatchewan_4.csv")
saskatchewan_5=pd.read_csv("saskatchewan_5.csv")
saskatchewan_6=pd.read_csv("saskatchewan_6.csv")


saskatchewan_all=pd.concat([saskatchewan_1,saskatchewan_2,saskatchewan_3,saskatchewan_4,saskatchewan_5,saskatchewan_6]).drop_duplicates()
saskatchewan_all['Province']="SK"
saskatchewan_all.to_csv("saskatchewan_all.csv", sep=',', encoding='utf-8')

##
montreal=pd.read_csv("montreal.csv")
montreal["Province"]="QC"
montreal.to_csv("montreal.csv", sep=',', encoding='utf-8')

## Concatenate all
merged_all=pd.concat([ns_all,alberta_all,saskatchewan_all,montreal]).drop_duplicates()
merged_all=merged_all.loc[:,['user', 'tweet-id', 'timestamp','likes','replies','retweets', 'text', 'Province']]



merged_all.columns


##

##
edmonton=pd.read_csv("edmonton.csv")
edmonton.columns
edmonton = edmonton.iloc[:,1:]

halifax=pd.read_csv("halifax.csv")
halifax = halifax.iloc[:,1:]

montreal_1=pd.read_csv("montreal.csv")
montreal_1 = montreal_1.iloc[:,1:]

regina=pd.read_csv("regina.csv")
regina = regina.iloc[:,1:]

toronto=pd.read_csv("toronto.csv")
toronto = toronto.iloc[:,1:]

vancouver=pd.read_csv("vancouver.csv")
vancouver = vancouver.iloc[:,1:]

winnipeg=pd.read_csv("winnipeg.csv")
winnipeg = winnipeg.iloc[:,1:]

merged_all_provinces=pd.concat([merged_all,edmonton,halifax,montreal_1,regina,toronto,vancouver,winnipeg]).drop_duplicates(subset=['tweet-id'])


# combine our canada


pre_legalization_all=pd.read_csv("pre_legalization_all.csv")
pre_legalization_all['Province']=np.NaN
canada_albert = pd.read_csv("canada.csv")
canada_albert = canada_albert.iloc[:,1:]

merged_all_canada=pd.concat([pre_legalization_all,canada_albert]).drop_duplicates(subset=['tweet-id'])


merged_final=pd.concat([merged_all_provinces,merged_all_canada]).drop_duplicates(subset=['tweet-id'])

merged_all_canada.to_csv("merged_all_canada.csv", sep=',', encoding='utf-8')
merged_all_provinces.to_csv("merged_all_provinces.csv", sep=',', encoding='utf-8')
merged_final.to_csv("merged_final.csv", sep=',', encoding='utf-8')



''' 
preprocessing

'''

''' 
Province Level Hashtags

'''

########### At each province level, find the hashtags that are the most repititive

provinces=pd.read_csv("merged_all_provinces.csv")

# comments per province
provinces.groupby(["Province"])['tweet-id'].count()

# HAshtags per province
#tokenize words
provinces["tokenized_text"] = provinces["text"].apply(nltk.word_tokenize)


#
#indices = [i for i, x in enumerate(provinces["tokenized_text"][0]) if x == "#"]
#indices=np.array(indices)
#indices=indices+1
#list(np.array(provinces["tokenized_text"][0])[indices])
#           

def find_hashtags(lst):
    list1=list()
    list2=list()
    indices = [i for i, x in enumerate(lst) if x == "#"]
    try :
        indices=np.array(indices)
        indices=indices+1
        list1=list(np.array(lst)[indices])
        for k in list1 : 
            list2.append(k.lower())
        return list2
    except:
        return list2

    
provinces["hashtags_word_tokenize"]=provinces["tokenized_text"].apply(lambda x :find_hashtags(x) )

# Find the frequency of hashtags per province

province_hash_freq =  collections.defaultdict(lambda : list())



for i in provinces["Province"]:
    lst=list()
    df = provinces[provinces["Province"]==i].loc[:,["hashtags_word_tokenize"]]
    for j in df["hashtags_word_tokenize"]:
        for k in j:
            lst.append(k)
    province_hash_freq[i]=lst
    
## COunt frequencies of hashtags
frequencies =  collections.defaultdict(lambda : list())
for i in province_hash_freq.keys():
    temp=nltk.FreqDist(tag for tag in province_hash_freq[i])
    frequencies[i]=temp.most_common(100)

frequencies_df=pd.DataFrame.from_dict(frequencies,orient='index').transpose()

''' 
	2. Find the frequencies of words at province level, excluding the hashtags

'''
wnl = nltk.WordNetLemmatizer()
def content_without_stopwords(lst):
    content = list()
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w.lower() for w in lst if (w.lower() not in stopwords)& (w.lower().isalpha())]
    lemmatized_attributes=[wnl.lemmatize(t) for t in content]
    return lemmatized_attributes

provinces["tokenized_text_without_stop"]=provinces["tokenized_text"].apply(lambda x :content_without_stopwords(x) )



# frequency of words without hashtags
i="SK"
province_without_hashtag =  collections.defaultdict(lambda : list())
for i in provinces["Province"]:
    lst=list()
    df = provinces[provinces["Province"]==i].loc[:,["tokenized_text_without_stop"]]
    for j in df["tokenized_text_without_stop"]:
        for k in j:
            if ((k not in province_hash_freq[i])):
                lst.append(k)
    province_without_hashtag["SK"]=lst

    
## COunt frequencies of hashtags
frequencies_without_hash =  collections.defaultdict(lambda : list())
for i in province_without_hashtag.keys():
    temp=nltk.FreqDist(tag for tag in province_without_hashtag[i])
    frequencies_without_hash[i]=temp.most_common(200)

frequencies_without_hash_df=pd.DataFrame.from_dict(frequencies_without_hash,orient='index').transpose()

frequencies_without_hash_df.to_csv("freq_without_hash.csv")
frequencies_df.to_csv("frequencies_df.csv")
''' 
Find the topics that were most commonly occuring at the province level. Can use topic modelling

'''


######################### FOR MY CODE
i=0

lst2=[]
for i in range(len(provinces["Province"])):
    pro=provinces.loc[i,"Province"]
    lst=provinces.loc[i,"tokenized_text_without_stop"]
    lst1=list()
    for j in lst:
        if ((j not in province_hash_freq[pro])):
            lst1.append(j)
    lst2.append(' '.join(lst1))
provinces["text_without_stop_without_hash"]=lst2
## 1sst method


reviews_df=provinces.loc[:,["Province","text_without_stop_without_hash"]]

#checking for nulls if present any
print("Number of rows with any of the empty columns:")
print(reviews_df.isnull().sum().sum())
reviews_df=reviews_df.dropna() 

restaurant_name = 'Province'
restaurant_review = 'text_without_stop_without_hash'
ntopics= 5

word_tokenizer=RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
stopwords_nltk=set(stopwords.words('english'))

def tokenize_text(version_desc):
    lowercase=version_desc.lower()
    text = wordnet_lemmatizer.lemmatize(lowercase)
    tokens = word_tokenizer.tokenize(text)
    return tokens

vec_words = CountVectorizer(tokenizer=tokenize_text,stop_words=stopwords_nltk,decode_error='ignore')
total_features_words = vec_words.fit_transform(reviews_df[restaurant_review])

print(total_features_words.shape)

model = lda.LDA(n_topics=int(ntopics), n_iter=500, random_state=1)
model.fit(total_features_words)

topic_word = model.topic_word_ 
doc_topic=model.doc_topic_
doc_topic=pd.DataFrame(doc_topic)
reviews_df=reviews_df.join(doc_topic)
restaurant=pd.DataFrame()

for i in range(int(ntopics)):
    topic="topic_"+str(i)
    restaurant[topic]=reviews_df.groupby([restaurant_name])[i].mean()
    
restaurant=restaurant.reset_index()
topics=pd.DataFrame(topic_word)
topics.columns=vec_words.get_feature_names()
topics1=topics.transpose()
topics1.to_excel("topic_word_dist.xlsx")
restaurant.to_excel("restaurant_topic_dist.xlsx",index=False)


''' 
4. Find the sentiments at each of the province (Pos, neg and neutral)

'''


#################################

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

# function to print sentiments 
# of the sentence. 
def sentiment_scores(sentence): 
	sid_obj = SentimentIntensityAnalyzer() 
	sentiment_dict = sid_obj.polarity_scores(sentence) 
	return sentiment_dict


neg=list()
neu=list()
pos=list()
compound=list()

text=list(provinces.text)
for i in text:
    senti_dict=sentiment_scores(i)
    neg.append(senti_dict["neg"])
    neu.append(senti_dict["neu"])
    pos.append(senti_dict["pos"])
    compound.append(senti_dict["compound"])

provinces['neg'] = neg
provinces['pos'] = pos
provinces['neu'] = neu
provinces['compound'] = compound

temp=provinces.describe()

provinces.groupby(['Province'])['neg'].mean()
provinces.groupby(['Province'])['pos'].mean()
provinces.groupby(['Province'])['neu'].mean()
provinces.groupby(['Province'])['compound'].mean()

''' 
5. Find the sentiments at each of the province (Pos, neg and neutral)-- different times

'''
provinces['timestamp'] = pd.to_datetime(provinces['timestamp'])

before_announcement=provinces[(provinces['timestamp'] < '2018-06-20 00:00:00') ]

announcement_legalization=provinces[(provinces['timestamp'] >= '2018-06-20 00:00:00') & (provinces['timestamp'] < '2018-10-17 00:00:00')]

after_legalization=provinces[(provinces['timestamp'] > '2018-10-17 00:00:00') ]

provinces.to_csv("Provinces_with_sentiment.csv")
### counts

before_announcement.groupby(['Province'])['Province'].count()
announcement_legalization.groupby(['Province'])['Province'].count()
after_legalization.groupby(['Province'])['Province'].count()

# overall positive/negative
before_announcement.groupby(['Province'])['compound'].mean()

announcement_legalization.groupby(['Province'])['compound'].mean()


after_legalization.groupby(['Province'])['compound'].mean()



# frequency of words without hashtags
i="SK"
provinces["hashtags_word_tokenize"]
province_without_hashtag_announcement_legalization =  collections.defaultdict(lambda : list())
for i in announcement_legalization["Province"]:
    lst=list()
    df = announcement_legalization[announcement_legalization["Province"]==i].loc[:,["hashtags_word_tokenize"]]
    for j in df["hashtags_word_tokenize"]:
        for k in j:
            lst.append(k)
    province_without_hashtag_announcement_legalization["SK"]=lst

    
## COunt frequencies of hashtags
frequencies_without_hash_announcement_legalization =  collections.defaultdict(lambda : list())
for i in province_without_hashtag_announcement_legalization.keys():
    temp=nltk.FreqDist(tag for tag in province_without_hashtag_announcement_legalization[i])
    frequencies_without_hash_announcement_legalization[i]=temp.most_common(200)

frequencies_without_hash_announcement_legalization_df=pd.DataFrame.from_dict(frequencies_without_hash_announcement_legalization,orient='index').transpose()


''' 
6.  Lets check Marijuana with other words in canada
'''


merged_all_canada=pd.read_csv("merged_all_canada.csv")

wnl = nltk.WordNetLemmatizer()
def content_without_stopwords(lst):
    content = list()
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w.lower() for w in lst if (w.lower() not in stopwords)& (w.lower().isalpha())]
    lemmatized_attributes=[wnl.lemmatize(t) for t in content]
    return lemmatized_attributes
merged_all_canada["text"].dropna(inplace=True)

merged_all_canada["tokenized_text"] = merged_all_canada["text"].apply(nltk.word_tokenize)
merged_all_canada["tokenized_text"].dropna(inplace=True)

merged_all_canada["tokenized_text_without_stop"]=merged_all_canada["tokenized_text"].apply(lambda x :content_without_stopwords(x) )

merged_all_canada["tokenized_text_without_stop"].dropna(inplace=True)

combined_tokens = list()

for i in merged_all_canada["tokenized_text_without_stop"]:
    for j in i: 
            combined_tokens.append(j.lower())

frequencies_combined=nltk.FreqDist(tag for tag in combined_tokens)
frequencies_combined.most_common(100)
all_freqs=dict(frequencies_combined)


temp=pd.DataFrame.from_dict(all_freqs,orient='index')


''' 
7.  Lets lift of words with each other--whichever occurs with marijuana

'''
list_keywords=["crimial","affect","problem","growth","culture","question","research","retail",
               "work","shortage","benefit","party","issue","stock","business","health",
               "tax","driving","trudeau","justin","enterpreneur","marijuanastocks","minister",
               "workplace","growing","doctor","senator", "challenge"]

list_keywords=["crimial"]
important_keywords = pd.concat([pd.Series(x) for x in list_keywords], axis=1)

#index the first row
important_keywords.columns = important_keywords.iloc[0]

#delete the first row
important_keywords=important_keywords[:-1]

#join two tables together

merged_all_canada = pd.concat([merged_all_canada, important_keywords], axis=1)



a=merged_all_canada["tokenized_text_without_stop"][0]

def find_word(lst):
    try : 
        if (word in lst):
            return True
        else:
            return False
    except : 
        return False

for word in list_keywords:
    merged_all_canada[word +""] = merged_all_canada["tokenized_text_without_stop"].apply(lambda x:find_word(x) )




merged_all_canada.to_csv("amanda.csv")

### Get cooccurance matrix.. 

#####################3

''' 
8.  Get the weighted sentiments and see if it is influenced by users with more followers

'''

#### Merge the table with Users's followers
result = pd.concat([before_announcement, users_followers], join='left',join_axes=[before_announcement.index])

before_announcement=pd.merge(before_announcement, users_followers, how='left')

before_announcement['followers'] = before_announcement['followers'].fillna(0)
before_announcement['neg'] = before_announcement['neg'].fillna(0)
before_announcement['pos'] = before_announcement['pos'].fillna(0)
before_announcement['neu'] = before_announcement['neu'].fillna(0)
before_announcement['compound'] = before_announcement['compound'].fillna(0)

before_announcement['neg_weighted']=before_announcement['neg']*np.log10(np.array((before_announcement['followers'])+0.00001))
before_announcement['pos_weighted']=before_announcement['pos']*np.log10(np.array((before_announcement['followers'])+0.00001))
before_announcement['neu_weighted']=before_announcement['neu']*np.log10(np.array((before_announcement['followers'])+0.00001))
before_announcement['compound_weighted']=before_announcement['compound']*np.log10(np.array((before_announcement['followers'])+0.00001))

before_announcement.groupby(['Province'])['compound_weighted'].mean()


####



announcement_legalization=pd.merge(announcement_legalization, users_followers, how='left')

announcement_legalization['followers'] = announcement_legalization['followers'].fillna(0)
announcement_legalization['neg'] = announcement_legalization['neg'].fillna(0)
announcement_legalization['pos'] = announcement_legalization['pos'].fillna(0)
announcement_legalization['neu'] = announcement_legalization['neu'].fillna(0)
announcement_legalization['compound'] = announcement_legalization['compound'].fillna(0)

announcement_legalization['neg_weighted']=announcement_legalization['neg']*np.log10(np.array((announcement_legalization['followers'])+0.00001))
announcement_legalization['pos_weighted']=announcement_legalization['pos']*np.log10(np.array((announcement_legalization['followers'])+0.00001))
announcement_legalization['neu_weighted']=announcement_legalization['neu']*np.log10(np.array((announcement_legalization['followers'])+0.00001))
announcement_legalization['compound_weighted']=announcement_legalization['compound']*np.log10(np.array((announcement_legalization['followers'])+0.00001))

announcement_legalization.groupby(['Province'])['compound_weighted'].mean()

###




after_legalization=pd.merge(after_legalization, users_followers, how='left')

after_legalization['followers'] = after_legalization['followers'].fillna(0)
after_legalization['neg'] = after_legalization['neg'].fillna(0)
after_legalization['pos'] = after_legalization['pos'].fillna(0)
after_legalization['neu'] = after_legalization['neu'].fillna(0)
after_legalization['compound'] = after_legalization['compound'].fillna(0)

after_legalization['neg_weighted']=after_legalization['neg']*np.log10(np.array((after_legalization['followers'])+0.00001))
after_legalization['pos_weighted']=after_legalization['pos']*np.log10(np.array((after_legalization['followers'])+0.00001))
after_legalization['neu_weighted']=after_legalization['neu']*np.log10(np.array((after_legalization['followers'])+0.00001))
after_legalization['compound_weighted']=after_legalization['compound']*np.log10(np.array((after_legalization['followers'])+0.00001))

after_legalization.groupby(['Province'])['compound_weighted'].mean()



''' 
9. Plot some keywords over time

'''

time_df = provinces.loc[:,['timestamp','tokenized_text_without_stop','Province']]
time_df.set_index(['just_date'])
time_df['timestamp']=pd.to_datetime(time_df['timestamp'])

time_df['just_date'] = time_df['timestamp'].dt.date

list_keywords=["criminal","affect","problem","growth","culture","question","research","retail",
               "work","shortage","benefit","party","issue","stock","business","health",
               "tax","driving","trudeau","justin","enterpreneur","marijuanastocks","minister",
               "workplace","growing","doctor","senator", "challenge"]


def find_word(lst):
    try : 
        if (word in lst):
            return True
        else:
            return False
    except : 
        return False

for word in list_keywords:
    time_df[word +""] = time_df["tokenized_text_without_stop"].apply(lambda x:find_word(x) )
    

mean_crimial = time_df.groupby(['just_date'])['criminal'].mean()
mean_crimial=mean_crimial.reset_index()
# Average of rstats column by day
mean_benefit =time_df.groupby(['just_date'])['benefit'].mean()
mean_benefit=mean_benefit.reset_index()
# Plot mean python by day(green)/mean rstats by day(blue)
plt.plot(mean_crimial.just_date, mean_crimial.criminal, color = 'green')
plt.plot(mean_benefit.just_date, mean_benefit.benefit, color = 'blue')

# Add labels and show
plt.xlabel('Year'); plt.ylabel('Proportion mentions')
plt.title('Keyword mentions over time')
plt.legend(('Criminal', 'Benifit'))
plt.show()
