
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class Kmeans:
    def __init__(self):
        self.all_movies = 'moviedata.csv'
        self.omdb_movies = 'ombddata.csv'

    def kmeanClustering(self):   
        le = LabelEncoder()
        ohe = OneHotEncoder() 
        df_movies = pd.read_csv(self.all_movies)
        df_omdb = pd.read_csv(self.omdb_movies)

        combined_df = pd.merge(df_movies, df_omdb, on="MovieId")
        combined_df['BoxOffice'] = combined_df['BoxOffice'].replace({'\$': '', ',': ''}, regex=True)
        combined_df['BoxOffice'].fillna(0,inplace=True)
        combined_df['Metascore'].fillna(0,inplace=True)

        combined_df['BoxOffice'] = combined_df['BoxOffice'].astype(int) 
        combined_df['Metascore'] = combined_df['Metascore'].astype(int) 

        combined_df = combined_df[combined_df['BoxOffice']>0]
        combined_df = combined_df[combined_df['Metascore']>0]

        selected_dataset = combined_df[['BoxOffice', 'Metascore']]
        print(selected_dataset)

        kmeans_object_Count = KMeans(n_clusters=3)
        kmeans_object_Count.fit(selected_dataset)
        prediction_kmeans = kmeans_object_Count.predict(selected_dataset)

        plt.scatter(combined_df['BoxOffice'], combined_df['Metascore'], c=prediction_kmeans, s=50, cmap='viridis')
        centers = kmeans_object_Count.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.xlabel("Box Office")
        plt.ylabel("MetaScore of the movie")
        plt.legend()
        plt.show()
        
classifier = Kmeans()
classifier.kmeanClustering()



