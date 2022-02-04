#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
r=requests.get("https://www.yelp.com/biz/vondas-kitchen-newark?osq=Restaurants")
r.status_code

#for div in divs: print(div)


# In[3]:


from bs4 import BeautifulSoup
soup =BeautifulSoup(r.text,'html.parser')
divs=soup.findAll(class_="comment__09f24__gu0rG css-1sufhje")
divs


# In[4]:


type(divs)
reviews=[]
for div in divs:
    reviews.append(div.text)


# # Data Analysis
# 

# In[5]:


import pandas as pd 
import numpy as np


# In[6]:


df=pd.DataFrame(np.array(reviews),columns=['review'])


# In[7]:


df.head()


# In[8]:


len(df.review)


# In[9]:


df.review.apply(lambda x: len(x.split()))


# In[10]:


df['word_count']=df.review.apply(lambda x: len(x.split()))


# In[11]:


df.head()


# In[12]:


df['char_count']=df.review.apply(lambda x: len(x))


# In[13]:


def avg(y):
    return (sum(len(x) for x in y.split())/len(y.split()))
   
df['avg_word_length']=df['review'].apply(lambda x: avg(x))
df.head()


# Importing stop words which don't hold importance to out analysis

# In[18]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=stopwords.words('english')


# In[19]:


df['stop_word_count']=df.review.apply(lambda x: len([word for word in x.split() if word.lower() in stop_words]))
df['stopword_rate']=df.stop_word_count/df.word_count


# In[20]:


df.head()


# In[21]:


df[df.stopword_rate>0.5]


# In[ ]:


df.sort_values(by="stopword_rate")


# # Data Clean up

# In[22]:


df.review=df.review.apply(lambda x: " ".join(word.lower() for word in x.split())) #LOWERCASE EVERYTHING


# In[28]:


df.head()


# In[29]:


df.review=df.review.str.replace('[^\w\s]','')#HERE WE TOOK CARE OF PUNCTUATIONS 


# In[31]:


df.review 


# In[47]:


df.review=df.review.apply(lambda x : ' '.join (word for word in x.split() if word not in stop_words))


# ### Now we will see those words which arent stop words but which dont contribute to the review. Hence we extract word of every review into a pandas series.

# In[50]:


pd.Series(' '.join(df['review']).split()).value_counts()[:50]


# In[52]:


other_words=['im','go','get','also']


# In[54]:


df['clean']=df.review.apply(lambda x: ' '.join(word for word in x.split() if word not in other_words))


# In[56]:


df.head()


# ## Lemmatization
# Condense the word back to their base format

# In[65]:


import import_ipynb
from textblob import Word
df['lemmatized']=df.clean.apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))


# In[67]:


df.head()


# # Sentiment Analysis
# TextBlob gives two values, polarity and subjectivity

# In[70]:


from textblob import TextBlob


# In[73]:


df['polarity']=df.lemmatized.apply(lambda x: TextBlob(x).sentiment[0])


# In[75]:


df['subjectivity']=df.lemmatized.apply(lambda x: TextBlob(x).sentiment[1])


# In[79]:


df.head()


# In[80]:


df.describe()


# In[ ]:




