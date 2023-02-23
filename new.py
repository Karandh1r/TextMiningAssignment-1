from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import pyLDAvis
import pyLDAvis.sklearn 
import matplotlib.pyplot as plt  
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
  
le = LabelEncoder()
ohe = OneHotEncoder()

file_name_movies = 'moviedata.csv'
file_movie_reviews = 'reviewdata.csv'

df_movie_reviews = pd.read_csv(file_movie_reviews)
print(df_movie_reviews.columns)

def lemmatize_words(text):
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

def stemming_words(text):
        ps = PorterStemmer()
        words = text.split()
        words = [ps.stem(word) for word in words]
        return ' '.join(words) 

df_movie_reviews['UserReviews'] = df_movie_reviews['UserReviews'].apply(lemmatize_words)
df_movie_reviews['UserReviews'] = df_movie_reviews['UserReviews'].apply(stemming_words) 

vectorizer = CountVectorizer(stop_words='english',max_features=50)
X = vectorizer.fit_transform(df_movie_reviews['UserReviews'])
column_names = vectorizer.get_feature_names_out()
count_vec_columns = column_names
vectorized_df = pd.DataFrame(X.toarray(),columns=column_names)
num_topics = 7
lda_model_DH = LatentDirichletAllocation(n_components=num_topics, 
                                         max_iter=100, learning_method='online')
LDA_DH_Model = lda_model_DH.fit_transform(vectorized_df)
print("SIZE: ", LDA_DH_Model.shape) 

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic:  ", idx)  
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

print_topics(lda_model_DH, vectorizer, 15)
word_topic = np.array(lda_model_DH.components_)
word_topic = word_topic.transpose()
num_top_words = 15
vocab_array = np.asarray(column_names)
pyLDAvis.sklearn.prepare(lda_model_DH, X, vectorizer)

fontsize_base = 20

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()

