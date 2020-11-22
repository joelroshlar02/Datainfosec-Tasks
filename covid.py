#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


os.chdir('C:\\Users\\mayur\\Downloads')


# In[3]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from os import path
from PIL import Image
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns


# In[76]:


data = pd.read_csv("covid19_tweets.csv")


# In[77]:


data


# In[78]:


data1=data.dropna()


# In[79]:


data1.shape


# In[81]:


data1.info()


# In[82]:


data1.describe()


# In[84]:


data1['text']=data1['text'].dropna() # For removing NA values


# In[85]:


data1['text']=data1['text'].str.lower() # To make sentence in lower case


# In[86]:


data1['text']=data1['text'].str.replace('http\S+|www.\S+','',case=False) # To remove http values


# In[87]:


data1['text']=data1['text'].str.replace('[^\w\s]','') # To remove Punctuation


# In[88]:


data1['text']=data1['text'].str.replace('\d+','') # To remove numeric values


# In[89]:


stop=set(stopwords.words('english')) 


# In[90]:


data1['text'].dropna(inplace=True)


# In[91]:


data1['text']=data1['text'].astype(str).apply(lambda line: [token for token in word_tokenize(line) if token not in stop ]) # To remove Stoopwords


# In[92]:


data1=data1['text'].dropna()


# In[93]:


data1


# In[94]:


st=PorterStemmer() # For word stemming


# In[95]:


data1=data1.apply(lambda x:[st.stem(y) for y in x])


# In[96]:


data1


# In[97]:


lemmatizer=WordNetLemmatizer() # For lemmatize


# In[98]:


data1=data1.apply(lambda x:[lemmatizer.lemmatize(y) for y in x])


# In[99]:


data1


# In[100]:


text=data1


# In[101]:


wordcloud=WordCloud(max_font_size=200,max_words=10000,background_color="whitesmoke").generate(str(text))


# In[102]:


plt.imshow(wordcloud,interpolation='nearest',aspect="auto")
plt.axis('off')
plt.figure(figsize=[20,10],facecolor='k')
plt.tight_layout(pad=0)
plt.show()


# In[124]:


mask=np.array(Image.open("m.png"))


# In[125]:


mask


# In[126]:


def transform_format(val):
    if val==0:
        return 255
    else:
        return val


# In[127]:


transformed_mask=np.ndarray((mask.shape[0],mask.shape[1]))

for i in range(len(mask)):
    transformed_mask[i]=list(map(transform_format,mask[i]))


# In[144]:


wc=WordCloud(background_color="white",max_words=1000,mask=transformed_mask,contour_width=1,contour_color='green')


# In[145]:


wc.generate(str(text))


# In[146]:


wc.to_file("m.png")


# In[149]:


plt.figure(figsize=[10,5])
plt.imshow(wc,interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[ ]:




