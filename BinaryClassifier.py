# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:15:41 2023

@author: 101908
"""

 # load text
filename = './data/01 - The Fellowship Of The Ring.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

words = text.split() # split into words by white space

words = [word.lower() for word in words] # remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
words = [w.translate(table) for w in words]
#print(stripped[:100])

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) 

filtered_words = [w for w in words if w not in stop_words]

from collections import Counter
 
# Pass the split_it list to instance of Counter class.
Counter = Counter(filtered_words)
  
# most_common() produces k frequently encountered
# input values and their respective counts.
most_occur = Counter.most_common(100)