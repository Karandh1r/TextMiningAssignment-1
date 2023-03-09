    
import nltk
from nltk.corpus import stopwords
import re
import requests
import json
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import graphviz
import numpy as np

all_news_data = []
file_name_news = 'exam_newsapidata.csv'
newscolumns = ['source','description','title','content','url']
topics = ["politics", "analytics", "business", "sports"]


def cleandescription(description):
    description = str(description)
    description = re.sub(r'[,.;@#?!&$\-\']+', ' ', description, flags=re.IGNORECASE)
    description = re.sub(' +', ' ', description, flags=re.IGNORECASE)
    description = re.sub(r'\"', ' ', description, flags=re.IGNORECASE)
    description = re.sub(r'[^a-zA-Z]', " ", description, flags=re.VERBOSE)
    description = description.replace(',', '')
    description = ' '.join(description.split())
    description = re.sub("\n|\r", "", description)
    description = ' '.join([wd for wd in description.split() if len(wd)>3])
    return description

def cleanTitle(Title):
    Title = str(Title)
    Title = re.sub(r'[,.;@#?!&$\-\']+', ' ', str(Title), flags=re.IGNORECASE)
    Title = re.sub(' +', ' ', str(Title), flags=re.IGNORECASE)
    Title = re.sub(r'\"', ' ', str(Title), flags=re.IGNORECASE)
    Title = re.sub(r'[^a-zA-Z]', " ", str(Title), flags=re.VERBOSE)
    Title = Title.replace(',', '')
    Title = ' '.join(Title.split())
    Title = re.sub("\n|\r", "", Title)
    return Title

def getDate(Date):
    NewDate = Date.split("T")
    Date = NewDate[0]
    return Date

def getdatafromnewApi():
    url = 'https://newsapi.org/v2/everything'
    apiKey = '120b448077954edebc2caa99382602cb'
    
    MyFILE=open(file_name_news, "a")
    WriteThis = 'LABEL,Date,Source,Title,Description\n'
    MyFILE.write(WriteThis)
    for topic in topics:
        try:
            params = {'q': topic, 
                    'apikey': apiKey}
            response = requests.get(url, params=params)  
            if response.status_code == 200:
                text_value = json.loads(response.text)
                all_articles = text_value['articles']
                for i in range(len(all_articles)):
                    movie_details = {}
                    if 'source' in all_articles[i]:
                        source = all_articles[i]["source"]["name"]
                    if 'description' in all_articles[i]:
                        filtered_description = cleandescription(all_articles[i]['description'])
                    if 'title' in all_articles[i]:
                        filtered_title = cleanTitle(all_articles[i]['title'])
                    if 'publishedAt' in all_articles[i]:
                        filtered_date = getDate(all_articles[i]['publishedAt'])
                    if 'content' in all_articles[i]:
                        movie_details['content'] = all_articles[i]['content']
                    WriteThis = str(topic)+","+str(filtered_date)+","+str(source)+","+ str(filtered_title) + "," + str(filtered_description) + "\n"   
                    MyFILE.write(WriteThis)                  
        except Exception as exp:
            print(f"error while hitting the news api",{exp})  
    MyFILE.close()        

getdatafromnewApi()

BBC_DF = pd.read_csv(file_name_news, error_bad_lines=False)
BBC_DF = BBC_DF.dropna()
HeadlineLIST = []
LabelLIST = []


for nexthead, nextlabel in zip(BBC_DF["Description"], BBC_DF["LABEL"]):
    HeadlineLIST.append(nexthead)
    LabelLIST.append(nextlabel)

NewHeadlineLIST=[]

for element in HeadlineLIST:
    AllWords = element.split(" ")
    #print(AllWords)
    NewWordsList=[]
    for word in AllWords:
        word = word.lower()
        if word in topics:
            continue
        else:
            NewWordsList.append(word)
    NewWords = " ".join(NewWordsList)
    NewHeadlineLIST.append(NewWords)

HeadlineLIST = NewHeadlineLIST
MyCountV = CountVectorizer(
        input="content",  ## because we have a csv file
        lowercase=True, 
        stop_words = "english",
        max_features=50
    )
MyDTM = MyCountV.fit_transform(HeadlineLIST) 
ColumnNames = MyCountV.get_feature_names()
MyDTM_DF = pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)
Labels_DF = DataFrame(LabelLIST,columns=['LABEL'])
My_Orig_DF = MyDTM_DF
dfs = [Labels_DF, MyDTM_DF]
Final_News_DF_Labeled = pd.concat(dfs,axis=1, join='inner')

List_of_WC=[]

for mytopic in topics:

    tempdf = Final_News_DF_Labeled[Final_News_DF_Labeled['LABEL'] == mytopic]    
    tempdf =tempdf.sum(axis=0,numeric_only=True)
    NextVarName=str("wc"+str(mytopic))
    NextVarName = WordCloud(width=1000, height=600, background_color="white",
                   min_word_length=4,
                   max_words=200).generate_from_frequencies(tempdf)
    List_of_WC.append(NextVarName)

fig = plt.figure(figsize=(25, 25))
NumTopics = len(topics)
for i in range(NumTopics):
    ax = fig.add_subplot(NumTopics,1,i+1)
    plt.imshow(List_of_WC[i], interpolation='bilinear')
    plt.axis("off")
    plt.savefig("NewClouds.pdf")

My_KMean = KMeans(n_clusters=3)
My_KMean.fit(My_Orig_DF)
My_labels = My_KMean.predict(My_Orig_DF)
print(My_labels)   

My_KMean2 = KMeans(n_clusters=4).fit(preprocessing.normalize(My_Orig_DF))
My_KMean2.fit(My_Orig_DF)
My_labels2=My_KMean2.predict(My_Orig_DF)
print(My_labels2)

My_KMean3= KMeans(n_clusters=3)
My_KMean3.fit(My_Orig_DF)
My_labels3=My_KMean3.predict(My_Orig_DF)
print("Silhouette Score for k = 3 \n",silhouette_score(My_Orig_DF, My_labels3))

cosdist = 1 - cosine_similarity(MyDTM)
print(cosdist)
print(np.round(cosdist,3))


linkage_matrix = ward(cosdist)  
print(linkage_matrix)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(linkage_matrix)
plt.show()


Final_News_DF_Labeled.to_csv("Labeled_News_Data_from_API.csv")
TrainDF, TestDF = train_test_split(Final_News_DF_Labeled, test_size=0.3)
### TEST----------------------
TestLabels = TestDF["LABEL"]
TestDF = TestDF.drop(["LABEL"], axis=1)
### TRAIN----------------------
TrainLabels = TrainDF["LABEL"]
TrainDF = TrainDF.drop(["LABEL"], axis=1)

##################################################
## STEP 3:  Run MNB
##################################################

## Instantiate
MyModelNB = MultinomialNB()

## FIT
MyNB = MyModelNB.fit(TrainDF, TrainLabels)
Prediction = MyModelNB.predict(TestDF)
print(np.round(MyModelNB.predict_proba(TestDF),2))

## COnfusion Matrix Accuracies
cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)


##################################################
## STEP 3:  Run DT
##################################################

## Instantiate
MyDT=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0,  
                            class_weight=None)

MyDT.fit(TrainDF, TrainLabels)


feature_names=TrainDF.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                      feature_names=feature_names,  
                      class_names=topics,  
                      filled=True, rounded=True,  
                      special_characters=True)      
                              
graph = graphviz.Source(Tree_Object) 
    
graph.render("MyTree") 


## COnfusion Matrix
print("Prediction\n")
print(TestDF)
DT_pred=MyDT.predict(TestDF)
print(DT_pred)
print(TestLabels)
    
bn_matrix = confusion_matrix(TestLabels, DT_pred)
print("\nThe confusion matrix is:")
print(bn_matrix)

######################### Plot Confusion matrix------------
import seaborn as sns
import matplotlib.pyplot as plt     

##########################################################
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


cm = confusion_matrix(TestLabels, DT_pred, labels=MyDT.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=MyDT.classes_)                         
disp.plot()
plt.show()

FeatureImp=MyDT.feature_importances_   
indices = np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
        print ("feature name: ", feature_names[indices[f]])

