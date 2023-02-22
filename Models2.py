import requests
import pandas as pd
import json
import random as rd
import re
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix    
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import tree
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
        df_reviews = pd.read_csv(self.filtered_reviews)
        x = df_reviews['UserReviews']
        y = df_reviews['Sentiment']
        
        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)
        vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',stop_words='english')
        vectorizer.fit_transform(df_reviews['UserReviews'])

        Train_X_Tfidf = vectorizer.transform(x_train)
        Test_X_Tfidf = vectorizer.transform(x_test)
        MyDT = DecisionTreeClassifier(criterion='entropy',
                            splitter='best',
                            max_depth=7, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            class_weight=None)
        
                  
        MyDT.fit(Train_X_Tfidf, y_train)
        
        tree.plot_tree(MyDT)
        feature_names = vectorizer.get_feature_names_out() 
        dot_data = tree.export_graphviz(MyDT, out_file=None,
                            feature_names= feature_names,   
                            filled=True, rounded=True,  
                            special_characters=True)                                    
        graph = graphviz.Source(dot_data) 
        tempname=str("Graph")
        graph.render(tempname) 
        
        print("Prediction\n")
        DT_pred=MyDT.predict(Test_X_Tfidf)
        print(DT_pred)
        print(MyDT.score(Test_X_Tfidf,y_test))
           
        bn_matrix = confusion_matrix(y_test, DT_pred)
        print("\nThe confusion matrix is:")
        print(bn_matrix)
        FeatureImp=MyDT.feature_importances_   
        indices = np.argsort(FeatureImp)[::-1]
          
        for f in range(x_train.shape[0]):
            if FeatureImp[indices[f]] > 0:
                print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
                print ("feature name: ", feature_names[indices[f]])

    def SVM(self):
        df_reviews = pd.read_csv(self.filtered_reviews)
        print(df_reviews.columns)
        x = df_reviews['UserReviews']
        y = df_reviews['Sentiment']
        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)
        SVM_Model = LinearSVC(C=1)
        
        vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',stop_words='english')
        vectorizer.fit_transform(df_reviews['UserReviews'])

        Train_X_Tfidf = vectorizer.transform(x_train)
        Test_X_Tfidf = vectorizer.transform(x_test)

                            
        SVM_Model.fit(Train_X_Tfidf, y_train)
        print("SVM prediction using Linear Kernel:\n", SVM_Model.predict(Test_X_Tfidf))
        print("SVM score using the Linear Kernel:\n",SVM_Model.score(Test_X_Tfidf,y_test))
        print("Actual:")
        print(y_test)
        SVM_matrix = confusion_matrix(y_test, SVM_Model.predict(Test_X_Tfidf))
        print("\nThe confusion matrix is:")
        print(SVM_matrix)
        print("\n\n")
 
        SVM_Model2 = sklearn.svm.SVC(C=50, kernel='rbf', 
                           verbose=True, gamma="auto")                    
        SVM_Model2.fit(Train_X_Tfidf, y_train)
        print("SVM prediction using Radial Basis function Kernel :\n", SVM_Model2.predict(Test_X_Tfidf))
        print("SVM score using the Radial Basis function Kernel:\n",SVM_Model2.score(Test_X_Tfidf,y_test))
        print("Actual:")
        print(y_test)
        print("RBF  :\n")
        SVM_matrix2 = confusion_matrix(y_test, SVM_Model2.predict(Test_X_Tfidf))
        print("\nThe confusion matrix is:")
        print(SVM_matrix2)
        print("\n\n")

        SVM_Model3=sklearn.svm.SVC(C=100, kernel='poly',degree=3,
                           gamma="auto", verbose=True)
        SVM_Model3.fit(Train_X_Tfidf, y_train)
        print("SVM prediction using Polynomial Kernel :\n",SVM_Model3.predict(Test_X_Tfidf))
        print("SVM score using Polynomial Kernel:\n",SVM_Model2.score(Test_X_Tfidf,y_test))
        print("Actual:")
        print(y_test)
        print("RBF  :\n")
        SVM_matrix3 = confusion_matrix(y_test, SVM_Model3.predict(Test_X_Tfidf))
        print("\nThe confusion matrix is:")
        print(SVM_matrix3)
        print("\n\n")

        top_features = 10
        COLNAMES = vectorizer.get_feature_names()
        coef = SVM_Model.coef_.ravel()
        top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
        top_positive_coefficients = top_positive_coefficients[top_positive_coefficients < 3451]

        top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
        top_negative_coefficients = top_negative_coefficients[top_negative_coefficients < 3451]

        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
        plt.bar(x=  np.arange(len(top_coefficients))  , height=coef[top_coefficients], width=.5,  color=colors)
        feature_names = np.array(COLNAMES)
        plt.xticks(np.arange(0, len(top_coefficients)), feature_names[top_coefficients], rotation=60, ha="right")
        plt.show()
        
     
ml_models = MLmodels()
# ml_models.labeldata()
#ml_models.NaiveBayes()
#ml_models.DecisionTreeClassifier()
ml_models.SVM()
