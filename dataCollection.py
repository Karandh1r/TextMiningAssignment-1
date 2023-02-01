from bs4 import BeautifulSoup 
import numpy as np
import pandas as pd
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import json
import requests
from tqdm import tqdm
from scrapy.selector import Selector
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
driver = webdriver.Chrome(ChromeDriverManager().install())

class WebScrapper:
    def __init__(self):
        self.url = 'https://www.imdb.com/chart/top/?ref_=nv_mv_250'
        self.parser = 'html.parser'
        self.header_list = ['Sno','MovieName','MovieRating','MovieUrl','MovieUserRating']
        self.prefix = 'https://www.imdb.com' 
        self.suffix = 'reviews/?ref_=tt_ql_urv'
        self.file_name_movies = 'moviedata.csv'
        self.file_name_reviews = 'reviewdata.csv'
        self.file_name_ombd = 'ombddata.csv'
        self.file_name_news = 'newsapidata.csv'
        self.columns = ['MovieId','MovieName','Rating','Url','UserRatings']
        self.omdbcolumns = ['MovieId','Title','Year','Rated','Released','Runtime','Genre','Director','Writer','Actors','Plot','Language','Country'
        'Awards','Metascore','imdbRating','imdbVotes','BoxOffice']
        self.newscolumns = ['MovieId','description','title','content','url']
        self.userreviewscolumns = ['MovieId','UserReviews']
        self.all_movies_reviews = []
        self.all_moview_ombd_reviews = []
        self.all_news_data = []

    def gettopMovies(self):  
        try:  
            response = requests.get(self.url) 
            data = BeautifulSoup(response.text,self.parser)
            all_movie_list = data.find_all('tr')
            header_list = self.header_list
            all_movies_reviews = []
            prefix = 'https://www.imdb.com'
            suffix = 'reviews/?ref_=tt_ql_urv'
            for movie in all_movie_list[1:21]:
                movie_review = {}
                movie_name = movie.find('td',{'class' : 'titleColumn'}).a.text
                movie_rating = movie.find('td',{'class' : 'ratingColumn'}).strong.text
                movie_url = movie.find('td',{'class' : 'titleColumn'}).a['href']
                movie_user_rating = movie.find('td',{'class': 'ratingColumn'}).strong['title']
                movie_review['MovieId'] = movie_url.split("/")[2]
                movie_review['MovieName'] = movie_name
                movie_review['Rating'] = movie_rating
                movie_review['Url'] = prefix + movie_url + suffix
                movie_review['UserRatings'] = movie_user_rating.split(" ")[3]
                self.all_movies_reviews.append(movie_review) 
            review_df = pd.DataFrame(self.all_movies_reviews,columns = self.columns)
            review_df.to_csv(self.file_name_movies,index=False) 
        except Exception as exp:
            print(f"error while retrieving top 25 movies webscraping {exp})")    
     
    def getuserReviews(self):
        for i in range(len(self.all_movies_reviews)):
            url = self.all_movies_reviews[i]['Url']
            movie_Id = self.all_movies_reviews[i]['MovieId']
            time.sleep(1)
            driver.get(url)
            time.sleep(1)
            print(driver.title)
            sel = Selector(text = driver.page_source)
            review_counts = sel.css('.lister .header span::text').extract_first().replace(',','').split(' ')[0]
            more_review_pages = int(int(review_counts)/25)
            for i in tqdm(range(more_review_pages)):
                try:
                    css_selector = 'load-more-trigger'
                    driver.find_element(By.ID, css_selector).click()
                except:
                    pass
            reviews = driver.find_elements(By.CSS_SELECTOR, 'div.review-container')
            first_review = reviews[0]
            sel2 = Selector(text = first_review.get_attribute('innerHTML'))
            review = sel2.css('.text.show-more__control::text').extract_first().strip()
            print('nreview:',review)
            review_list = []
            error_url_list = []
            error_msg_list = []
            movie_id_list = []
            reviews = driver.find_elements(By.CSS_SELECTOR, 'div.review-container')
            for d in tqdm(reviews):
                try:
                    sel2 = Selector(text = d.get_attribute('innerHTML'))
                    try:
                        review = sel2.css('.text.show-more__control::text').extract_first()
                        review_list.append(review)
                        movie_id_list.append(movie_Id)
                    except:
                        review = np.NaN
                        review_list.append(review)
                except Exception as e:
                    error_url_list.append(url)
                    error_msg_list.append(e)
            review_df = pd.DataFrame({'MovieId' : movie_id_list,'UserReviews':review_list})
            review_df.to_csv(self.file_name_reviews, mode='a', index=False,header=False) 
        review_df.to_csv(self.file_name_reviews, mode='a',header=self.userreviewscolumns, index=False)     

    def openmoviedatabaseApi(self):
        url = 'http://www.omdbapi.com/'
        apiKey = '7d04b063'
        all_moview_ombd_reviews = []
        for i in range(len(self.all_movies_reviews)):
            movie_string = self.all_movies_reviews[i]['MovieName'].replace(" ","+")
            movie_id = self.all_movies_reviews[i]['MovieId']
            params = {'t': movie_string, 
                    'apikey': apiKey}
            movie_details = {}           
            try:           
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    text_value = json.loads(response.text)
                    movie_details['MovieId'] = movie_id  
                    if 'Title' in text_value:
                        movie_details['Title'] = text_value['Title']
                    if 'Year' in text_value:    
                        movie_details['Year'] = text_value['Year']
                    if 'Rated' in text_value:
                        movie_details['Rated'] = text_value['Rated']
                    if 'Released' in text_value:
                        movie_details['Released'] = text_value['Released']
                    if 'Runtime' in text_value:
                        movie_details['Runtime'] = text_value['Runtime']
                    if 'Genre' in text_value:
                        movie_details['Genre'] = text_value['Genre']
                    if 'Director' in text_value:
                        movie_details['Director'] = text_value['Director']
                    if 'Writer' in text_value:
                        movie_details['Writer'] = text_value['Writer']
                    if 'Actors' in text_value:
                        movie_details['Actors'] = text_value['Actors']
                    if 'Plot' in text_value:
                        movie_details['Plot'] = text_value['Plot']
                    if 'Language' in text_value:
                        movie_details['Language'] = text_value['Language']
                    if 'Country' in text_value:
                        movie_details['Country'] = text_value['Country']
                    if 'Awards' in text_value:
                        movie_details['Awards'] = text_value['Awards']
                    if 'Metascore' in text_value:
                        movie_details['Metascore'] = text_value['Metascore']
                    if 'imdbRating' in text_value:
                        movie_details['imdbRating'] = text_value['imdbRating']
                    if 'imdbVotes' in text_value:
                        movie_details['imdbVotes'] = text_value['imdbVotes']    
                    if 'BoxOffice' in text_value:
                        movie_details['BoxOffice'] = text_value['BoxOffice']   
                    self.all_moview_ombd_reviews.append(movie_details)     
            except Exception as exp:
                print(f"error while hitting the omdb api request {exp})")  
        ombd_df = pd.DataFrame(self.all_moview_ombd_reviews,columns = self.omdbcolumns)
        ombd_df.to_csv(self.file_name_ombd,index=False) 

    def getdatafromnewApi(self):
        url = 'https://newsapi.org/v2/everything'
        apiKey = '120b448077954edebc2caa99382602cb'
        for i in range(len(self.all_movies_reviews)):
            try:
                movie_string =  self.all_movies_reviews[i]['MovieName']
                movie_id = self.all_movies_reviews[i]['MovieId']
                params = {'q': movie_string, 
                    'apikey': apiKey}
                response = requests.get(url, params=params)
                movie_details = {}  
                if response.status_code == 200:
                    text_value = json.loads(response.text)
                    all_articles = text_value['articles']
                    movie_details['MovieId'] = movie_id 
                    for i in range(len(all_articles)):
                        if 'description' in all_articles[i]:
                            movie_details['description'] = all_articles[i]['description']
                        if 'title' in all_articles[i]:
                            movie_details['title'] = all_articles[i]['title']
                        if 'content' in all_articles[i]:
                            movie_details['content'] = all_articles[i]['content']
                        if  'url' in all_articles[i]:
                            movie_details['url'] = all_articles[i]['url']
                self.all_news_data.append(movie_details)
                news_df = pd.DataFrame(self.all_news_data)
                news_df.to_csv(self.file_name_news, mode='a', index=False,header=False)

            except Exception as exp:
                print(f"error while hitting the news api",{exp})    
        #news_df.to_csv(self.file_name_news, mode='a',header=self.newscolumns, index=False)          

web_scrap = WebScrapper()      
web_scrap.gettopMovies()
web_scrap.getuserReviews()      
web_scrap.openmoviedatabaseApi()
web_scrap.getdatafromnewApi()



