"""


Text Classification Using Python and Scikit-learn

https://dylancastillo.co/text-classification-using-python-and-scikit-learn/


"""
 

 
import pandas as pd

from sklearn.datasets import fetch_20newsgroups

from SupervisedLearning.SupervisedLearning import SupvLrn
 
categories = [
    "alt.atheism",
    "misc.forsale",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
]

news_group_data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), categories=categories)

df = pd.DataFrame(dict(text=news_group_data["data"], target=news_group_data["target"]))
df["target"] = df.target.map(lambda x: categories[x])

txtclassmodel = SupvLrn.TextClassification(df)
text_class = txtclassmodel.predict_text_class(["Don't take it away! I will pull the trigger", "Jesus lives!"])
 















