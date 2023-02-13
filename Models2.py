import requests
import pandas as pd
import json
import random as rd
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix    
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import graphviz 
from sklearn.svm import LinearSVC   
from sklearn.svm import SVC

class MLmodels:
    def __init__(self):
        self.url = 'https://text-analysis12.p.rapidapi.com/sentiment-analysis/api/v1.1'
        self.apiKey = '117cd9915amsh12494a2bdb16d7ep13b3fdjsnd8861fc6f602'
        self.filtered_reviews = 'filtered.csv'
        self.file_name_reviews = 'reviewdata.csv'
        self.all_movies_reviews = []
        self.userreviewscolumns = ['MovieId','UserReviews','Sentiment']
        

    def labeldata(self):
        df_reviews = pd.read_csv(self.file_name_reviews)
        for idx in range(len(df_reviews[1:300])):
            movie_details_dict = {}
            text_value = df_reviews['UserReviews'][idx]
            movie_id = df_reviews['MovieId'][idx]
            try:
                payload = {
                    "language": "english",
                    "text": text_value
                }
                headers = {
                    "content-type": "application/json",
                    "X-RapidAPI-Key": self.apiKey,
                    "X-RapidAPI-Host": "text-analysis12.p.rapidapi.com"
                }    
                response = requests.request("POST", self.url, json=payload, headers=headers)
                txt = json.loads(response.text)
                movie_details_dict['MovieId'] = movie_id
                movie_details_dict['UserReviews'] = text_value
                movie_details_dict['Sentiment'] = txt["sentiment"]
                self.all_movies_reviews.append(movie_details_dict)
            except Exception as exp:
                print(f"error occured while labelling of the data. {exp})")  
        filtered_df = pd.DataFrame(self.all_movies_reviews,columns = self.userreviewscolumns)
        filtered_df.to_csv(self.filtered_reviews,index=False)    

    def NaiveBayes(self):
        df_reviews = pd.read_csv(self.filtered_reviews)
        print(df_reviews.columns)
        x = df_reviews['UserReviews']
        y = df_reviews['Sentiment']
        x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)
        vec = CountVectorizer(stop_words='english')
        x = vec.fit_transform(x).toarray()
        x_test = vec.transform(x_test).toarray() 
        model = MultinomialNB()
        model.fit(x, y)       
        y_predict = model.predict(x_test)
        print(model.score(x_test, y_test))
        cnf_matrix1 = confusion_matrix(y_test, y_predict)
        print("\nThe confusion matrix is:")
        print(cnf_matrix1)  

    def DecisionTreeClassifier(self):
        MyDT = DecisionTreeClassifier(criterion='entropy',
                            splitter='best',
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)

    def SVM(self):
        df_reviews = pd.read_csv(self.filtered_reviews)
        print(df_reviews.columns)
        x = df_reviews['UserReviews']
        y = df_reviews['Sentiment']
        x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)
        SVM_Model = LinearSVC(C=1)
        vec = CountVectorizer(stop_words='english')
        x = vec.fit_transform(x).toarray()
        x_test = vec.transform(x_test).toarray() 
        SVM_Model.fit(x, y)
        self.plot_coefficients(SVM_Model, vec.get_feature_names())
        print("SVM prediction using Linear Kernel:\n", SVM_Model.predict(x_test))
        print("SVM score using the Linear Kernel:\n",SVM_Model.score(x_test,y_test))
        print("Actual:")
        print(y_test)
        SVM_matrix = confusion_matrix(y_test, SVM_Model.predict(x_test))
        print("\nThe confusion matrix is:")
        print(SVM_matrix)
        print("\n\n")
 
        SVM_Model2 = sklearn.svm.SVC(C=50, kernel='rbf', 
                           verbose=True, gamma="auto")                    
        SVM_Model2.fit(x, y)
        print("SVM prediction using Radial Basis function Kernel :\n", SVM_Model2.predict(x_test))
        print("SVM score using the Radial Basis function Kernel:\n",SVM_Model2.score(x_test,y_test))
        print("Actual:")
        print(y_test)
        print("RBF  :\n")
        SVM_matrix2 = confusion_matrix(y_test, SVM_Model2.predict(x_test))
        print("\nThe confusion matrix is:")
        print(SVM_matrix2)
        print("\n\n")

        SVM_Model3=sklearn.svm.SVC(C=100, kernel='poly',degree=3,
                           gamma="auto", verbose=True)
        SVM_Model3.fit(x, y)
        print("SVM prediction using Polynomial Kernel :\n",SVM_Model3.predict(x_test))
        print("SVM score using Polynomial Kernel:\n",SVM_Model2.score(x_test,y_test))
        print("Actual:")
        print(y_test)
        print("RBF  :\n")
        SVM_matrix3 = confusion_matrix(y_test, SVM_Model3.predict(x_test))
        print("\nThe confusion matrix is:")
        print(SVM_matrix3)
        print("\n\n")

    def plot_coefficients(classifier, feature_names, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
        coef = classifier.coef_.ravel()
        top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
        print(top_positive_coefficients)
        top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
        print(top_negative_coefficients)
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
        plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
        plt.show()
    

 
ml_models = MLmodels()
# ml_models.labeldata()
#ml_models.NaiveBayes()
#ml_models.DecisionTreeClassifier()
ml_models.SVM()
ml_models.plot_coefficients()
