import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import CountVectorizer, TfidVectorizer
#from sklearn.metrics import f1_score

# =============================================================================
# 
#  Machine Learning Process:
#
#  - What question are we trying to answer?
#     ---> Find data to help answer question
#          ---> Process data
#                 ---> Build Model
#                     ---> Evaluate Model
#                          ----> Improve Model Further
# 
#   Machine Learning has two kinds of models:
#    1. Traditional, Algorithmic Models  (Sci-kit learn is mostly this)
#    2. Neural Network Models (Tensorflow, Pytorch)
#
# =============================================================================

import json

filename = "./Books_small.json"

class Sentiment:
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
        
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else: 
            return Sentiment.POSITIVE



reviews = []
with open(filename) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

 


training, test = train_test_split(reviews, test_size=0.33, random_state=42)



train_x = [x.text for x in training]
train_y = [x.sentiment for x in training]

test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]


# =============================================================================
# Bag of words vectorization
# 
# =============================================================================

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)


from sklearn import svm

cls_svm = svm.SVC(kernel='linear')
cls_svm.fit(train_x_vectors, train_y)

p = cls_svm.predict(test_x_vectors[0])
print(p)

# Lets calculate how many predictions we get correct:

modelscore = cls_svm.score(test_x_vectors, test_y)

print(modelscore)

# =============================================================================
# 
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
clf_def = DecisionTreeClassifier()


# =============================================================================
# class Category:
#     ELECTRONICS = "ELECTRONICS"
#     BOOKS = "BOOKS"
#     
# 
# =============================================================================
    