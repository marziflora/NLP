#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from sqlalchemy import create_engine
import psycopg2
import config
import pandas as pd
import os
from datetime import date
import numpy as np
from datetime import datetime
import dateutil
import datetime
import dateutil.relativedelta
import numpy as np; np.random.seed(0)
import seaborn as sns

f = open(path_to_password, "r")
password = f.read()

conn = psycopg2.connect(
    host=host,
    database=database,
    user=user,
    password=password)
conn.autocommit = True
get_ipython().run_line_magic('reload_ext', 'sql_magic')


# In[ ]:


get_ipython().run_cell_magic('read_sql', 'df -c conn', 'SELECT DISTINCT idd_czynnosci, nazwaczynnosci FROM table_to_analyse;')


# In[ ]:


raw_text = df.nazwaczynnosci.values
raw_text[5]


# ## Lower text 

# In[ ]:


clean_text1 = []

def lower_case(data):
    for words in raw_text:
        clean_text1.append(str.lower(words))
lower_case(raw_text)
clean_text1[5] 


# In[ ]:


# !pip install nltk


# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize #split into sentences 
import nltk
# nltk.download('punkt')


# ## 1-gram

# In[ ]:


clean_text2 = [] 

for sentence in raw_text:
    clean_text2.extend(sentence.split(" "))


# In[ ]:


from collections import Counter

count_dictionary = Counter(clean_text2)
sorted_dictionary = {k: [v] for k, v in sorted(count_dictionary.items(), key=lambda item: item[1], reverse=True)}


# In[ ]:


df_cnt1 = pd.DataFrame(sorted_dictionary)
df_cnt1 = df_cnt1.T
df_cnt1 = df_cnt1.reset_index()
df_cnt1.columns = ['Słowo', 'Cnt']
df_cnt1.head()


# In[ ]:


df_cnt1.to_csv("1gram.csv", index=False, encoding='utf-8-sig', sep="\t")


# ## Delete special signs  [](),-.:=>/_+ 

# In[ ]:


clean_text2 = []

for words in clean_text1:
    w = re.sub("[\[\](),-.:=>/_+`]", " ", words)
    clean_text2.append(w)


# ## Connect special words

# In[ ]:


clean_text3 = []

for words in clean_text2:
    w = re.sub(" e kir", " e-kir", words)
    w = re.sub(" s a[ $]", " sa ", w)
    w = re.sub(" a s[ $]", " as ", w)
    w = re.sub(" z o[ $]", " zo ", w)
    w = re.sub(" o o[ $]", " oo ", w)
#     clean_text3.append(w)
#     if not w==words:
#         print(words, w)
#         print("==============================")
    clean_text3.append(w)


# ## n-gram

# In[ ]:


from nltk import bigrams
from nltk import ngrams

def create_ngram(n):
    count_dictionary2 = {}
    for line in clean_text3:
        try:
            token = nltk.word_tokenize(line)
        except:
            print(line)
        bigram = list(ngrams(token, n)) 
        fdist = nltk.FreqDist(bigram)
        for i,j in fdist.items():
            key = " ".join(i)
            if key in count_dictionary2.keys():
                count_dictionary2[key] += j #add how many times in this record 
            else:
                count_dictionary2[key] = j

    df_cnt = pd.DataFrame(count_dictionary2, index=[0])
    df_cnt = df_cnt.T
    df_cnt = df_cnt.reset_index()
    df_cnt.columns = ['Słowa', 'Cnt']
    df_cnt = df_cnt.sort_values(by='Cnt', ascending=False)
    df_cnt.to_csv(f"{n}-grams.csv", index=False, sep="\t", encoding='utf-8-sig')
    return df_cnt

# df_cnt = create_ngram(2)
df_cnt = create_ngram(2)
df_cnt = create_ngram(3)
df_cnt = create_ngram(4)
df_cnt = create_ngram(5)
# df_cnt.head()


# In[ ]:


df_cnt


# In[ ]:





# In[ ]:




