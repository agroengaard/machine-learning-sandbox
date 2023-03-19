 

import re
import string
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB 
 
# =============================================================================
# 
# See SHAP
#
#
#
#
# =============================================================================



# =============================================================================
# 
# =============================================================================




class SupvLrn:
    """
    ===========================================================================
    || CLASS CONTAINING SUBCLASSES AND FUNCTIONS FOR SUPERVISED LEARNING     ||
    || ALGORITHMS                                                            ||
    ===========================================================================
    """
    
# =============================================================================
#     @staticmethod
#     def explain_the_model(model, X_train, x_test):
#         """
#         https://towardsdatascience.com/shap-explain-any-machine-learning-model-in-python-24207127cad7
#         https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137
#         """
#         import shap
#     #    explainer = shap.Explainer(model.predict, x_test)
#    #     shap_values = explainer(x_test)
#         explainer = shap.Explainer(model.predict, X_train)
#         shap_values = explainer.shap_values(X_train)
#         
#         shap.plots.waterfall(shap_values[0])
#         shap.plots.waterfall(shap_values[1])
#         shap.plots.bar(shap.Explanation(shap_values[1], feature_names=x_test.columns))
#         shap.summary_plot(shap_values)
# =============================================================================
    
    @staticmethod
    def process_text(text):
        """
        -----------------------------------------------------------------------
        | Function for cleaning your text so its ready for vectorization      |
        | Example of use:                                                     |
        |     df["clean_text"] = df.text.map(process_text)                    |
        -----------------------------------------------------------------------
        """
        text = str(text).lower()
        text = re.sub(
            f"[{re.escape(string.punctuation)}]", " ", text
        )
        text = " ".join(text.split())
        return text
    
 
    class TextClassification: 
        """
        -----------------------------------------------------------------------
        | Text Classification Using Python and Scikit-learn
        | I used this example for getting started, and the built from that:
        | https://dylancastillo.co/text-classification-using-python-and-scikit-learn/
        |----------------------------------------------------------------------
        """
        def __init__(self, df):
            
            print("Initializing Text Classification Model")
            print("")
            df["clean_text"] = df.text.map(SupvLrn.process_text)
            df_train, df_test = train_test_split(df, test_size=0.20, stratify=df.target) # Splitting our data into training data and test data
  
            self.vec = CountVectorizer(                                        # CountVectorizer turns text into numerical features.
                ngram_range=(1, 3), 
                stop_words="english",                                          # stop_words is a list of words that the function will ignore. In this case, the list "english" means that the function will ignore the most common words in English.
            )   

            X_train = self.vec.fit_transform(df_train.clean_text)              # Generating the matrices of token counts for the training set
            X_test = self.vec.transform(df_test.clean_text)                    # Generating the matrices of token counts for the test set 
 
            y_train = df_train.target                                          # Save the response variable from the training and testing set into y_train and y_test.
            y_test = df_test.target
 
            self.ml_model = SupvLrn.TextClassification.find_best_model(X_train, y_train, X_test, y_test)
            print("Model is: "+ self.ml_model.__class__.__name__)
          # SupvLrn.explain_the_model(self.ml_model, X_train, X_test)
 
# =============================================================================
#     
#             def predict_proba(X):
#                return self.ml_model.predict_proba(X)
#     
#             import shap
#         #    explainer = shap.Explainer(model.predict, x_test)
#        #     shap_values = explainer(x_test)
#           #  explainer = shap.Explainer(self.ml_model, X_train)
#             explainer = shap.KernelExplainer(predict_proba, shap.sample(X_train, 100))
#             shap_values = explainer.shap_values(shap.sample(X_train, 100))
#             
#          #   shap.plots.waterfall(shap_values[0])
#          #   shap.plots.waterfall(shap_values[1])
#          #   shap.plots.bar(shap.Explanation(shap_values[1], feature_names=X_test.columns))
#          #   shap.summary_plot(shap_values)
#             shap_display = shap.plots.beeswarm(shap_values, matplotlib=True)
#        #     display(shap_display)
#         
# =============================================================================
        @staticmethod
        def find_best_model(X_train, y_train, X_test, y_test):
            """
            -------------------------------------------------------------------
            | Finding the classification model with the best fit
            | From models that are commonly used in cases with discrete 
            | features such as word counts.
            |
            | See also:
            | https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
            | https://scikit-learn.org/stable/model_selection.html
            |__________________________________________________________________
            """
         #   models = ["MultinomialNB", "GaussianNB", "ComplementNB", "BernoulliNB", "CategoricalNB"]
         
            test_scores = dict()
                    
            best_model = None
            old_modelscore = 0
            
            for temp_ml_model in [MultinomialNB(),  ComplementNB(), BernoulliNB() ]:
                
                model_name = temp_ml_model.__class__.__name__
                
                print("-"*20+"Testing model: "+model_name+"-"*20)
                
                temp_ml_model.fit(X_train, y_train)
                preds = temp_ml_model.predict(X_test)  
                
                print(classification_report(y_test, preds))
                
                new_modelscore = temp_ml_model.score(X_test, y_test)
                test_scores[model_name] = new_modelscore
                
                if new_modelscore > old_modelscore:
                    best_model = temp_ml_model
                    
                old_modelscore = new_modelscore
                
            print(" ")
            print("="*25+" Test result is: "+"="*25)
            print(" ")
            
            for k,v in test_scores.items():
                print(k+ ": "+ str(v))
    
            max_key, max_value = max(test_scores.items(), key=lambda x: x[1])
            print(f"The best model was {max_key}, with accuracy {max_value}.")
            print("")
            
            
            
            return best_model
            
        
        def predict_text_class(self, sample_text):
            """
            -------------------------------------------------------------------
            |  Function for predicting classes/categories for string input    |
            |  based on the trained model                                     |
            -------------------------------------------------------------------
            |  INPUT:                                                         |
            |      sample_text (list) : List of strings to predict the class  |
            |                           for, for example:                     |
            |                          ["Jesus lives!", Don't take my guns!'] |
            |  RETURNS:                                                       |
            |      class_prediction (Array of str) : An array containing the  |
            |              predicted class/category for each string in the    |
            |              input list                                         |
            |_________________________________________________________________|
            """
            
         
            
            clean_sample_text = SupvLrn.process_text(sample_text)              # Process the text in the same way you did when you trained it!
            sample_vec = self.vec.transform(sample_text)
            class_prediction = self.ml_model.predict(sample_vec)
            
            print("Predicted text classes for: "+ str(len(sample_text)) +" string inputs")
            
            return class_prediction



# =============================================================================
# 
# =============================================================================

if __name__ == "__main__":
     
    import pandas as pd
    
    from sklearn.datasets import fetch_20newsgroups
    
 
     
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
     
    
    
    
    
    
    
    


