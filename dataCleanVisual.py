import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt 
from wordcloud import WordCloud

class CleanData:
    def __init__(self) -> None:
        self.file_name_movies = 'moviedata.csv'
        self.file_name_reviews = 'reviewdata.csv'
        self.file_name_ombd = 'ombddata.csv'
        self.file_name_news = 'newsapidata.csv'
        self.userreviewscolumns = ['MovieId','UserReviews']
        self.newscolumns = ['MovieId','description','title','content','url']
        self.visualizetfidfcolumns = []
        self.countvectorisecolumns = []

    def countVectoriser(self,df_news,search) -> None:
        vectorizer = CountVectorizer(stop_words='english',max_features=50)
        X = vectorizer.fit_transform(df_news[search])
        column_names = vectorizer.get_feature_names_out()
        self.countvectorisecolumns = column_names
        vectorized_df = pd.DataFrame(X.toarray(),columns=column_names)
        print(vectorized_df)
    
    def stemming_tokenizer(self,str_input):
        words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
        return words

    def tfidfvectorizer(self,df_news,search) -> None:
        vectorizer = TfidfVectorizer(stop_words='english', tokenizer = self.stemming_tokenizer,max_features=1000)
        X = vectorizer.fit_transform(df_news[search])
        column_names = vectorizer.get_feature_names_out()
        self.visualizetfidfcolumns = column_names
        vectorized_text = pd.DataFrame(X.toarray(),columns=column_names)
        print(vectorized_text)
    
    def lemmatize_words(self,text):
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def stemming_words(self,text):
        ps = PorterStemmer()
        words = text.split()
        words = [ps.stem(word) for word in words]
        return ' '.join(words)


    def cleanExceldata(self) -> None:
        df_movies = pd.read_csv(self.file_name_movies)
        df_reviews = pd.read_csv(self.file_name_reviews)
        df_ombd = pd.read_csv(self.file_name_ombd)
        df_news = pd.read_csv(self.file_name_news)

        df_news['description'] = df_news['description'].apply(self.lemmatize_words)
        df_news['description'] = df_news['description'].apply(self.stemming_words)
        self.countVectoriser(df_news,'description')
        self.tfidfvectorizer(df_news,'description')

        df_reviews['UserReviews'] = df_reviews['UserReviews'].apply(self.lemmatize_words)
        df_reviews['UserReviews'] = df_reviews['UserReviews'].apply(self.stemming_words)
        self.countVectoriser(df_reviews,'UserReviews')
        self.tfidfvectorizer(df_reviews,'UserReviews')

    def visualizeCountVect(self) -> None:
        text = " ".join(title for title in self.countvectorisecolumns)
        word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def visualizeTfidfVect(self) -> None:
        text = " ".join(title for title in self.visualizetfidfcolumns)
        word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


clean_data = CleanData()
clean_data.cleanExceldata()
clean_data.visualizeCountVect()
clean_data.visualizeTfidfVect()

         