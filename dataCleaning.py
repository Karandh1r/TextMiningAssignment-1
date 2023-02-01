import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re


class CleanData:
    def __init__(self) -> None:
        self.file_name_movies = 'moviedata.csv'
        self.file_name_reviews = 'reviewdata.csv'
        self.file_name_ombd = 'ombddata.csv'
        self.file_name_news = 'newsdata.csv'
        self.userreviewscolumns = ['MovieId','UserReviews']
        self.newscolumns = ['MovieId','description','title','content','url']
    def cleanExceldata(self) -> None:
        df_movies = pd.read_csv(self.file_name_movies, index_col=0)
        df_reviews = pd.read_csv(self.file_name_reviews,index_col=0)
        df_ombd = pd.read_csv(self.file_name_ombd,index_col=0)
        #df_news = pd.read_csv(self.file_name_news,index_col=0)
       

        #df_news.to_csv(self.file_name_news, header=self.newscolumns, index=False)
        df_reviews.to_csv(self.file_name_reviews,header=self.userreviewscolumns,index=False)

        stop_words = stopwords.words('english')
        #clean all customer reviews from the data.
        df_reviews['UserReviews'] = df_reviews['UserReviews'].str.extract('([A-Za-z]+)', expand=True)
        df_reviews['UserReviews'] = df_reviews['UserReviews'].str.extract('(\d+.\d)', expand=True)
        df_reviews['UserReviews'] = df_reviews['UserReviews'].apply(str.lower)
        df_reviews['UserReviews'] = df_reviews['UserReviews'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        df_reviews['UserReviews'] = df_reviews['UserReviews'].apply(lambda x: ' '.join(w for w in x.split() if w not in stop_words))

        
        # df_reviews['UserReviews'] = df_reviews['UserReviews'].str.extract('([A-Za-z]+)', expand=True)
        # df_reviews['UserReviews'] = df_reviews['UserReviews'].str.extract('(\d+.\d)', expand=True)
        # df_reviews['UserReviews'] = df_reviews['UserReviews'].apply(str.lower)
        # df_reviews['UserReviews'] = df_reviews['UserReviews'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        # df_reviews['UserReviews'] = df_reviews['UserReviews'].apply(lambda x: ' '.join(w for w in x.split() if w not in stop_words))

        print(df_movies.describe())

clean_data = CleanData()
clean_data.cleanExceldata()

         