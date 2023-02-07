import requests
from bs4 import BeautifulSoup 

url = 'https://www.imdb.com/search/title/?title_type=feature&year=2020-01-01,2020-12-31&sort=year,asc'
response = requests.get(url) 
data = BeautifulSoup(response.text,'html.parser')
all_movie_list = data.find_all('div',{'class' : 'lister-item mode-advanced'})
all_movies_reviews = []
prefix = 'https://www.imdb.com'
suffix = 'reviews/?ref_=tt_ql_urv'
for movie in all_movie_list[1:21]:
    movie_review = {}
    movie_name = movie.find('div',{'class' : 'lister-item-content'}).h3.a.text
    movie_url = movie.find('div',{'class' : 'lister-item-content'}).a['href']
    movie_review['MovieId'] = movie_url.split("/")[2]
    movie_review['MovieName'] = movie_name
    movie_review['Url'] = prefix + movie_url + suffix
    all_movies_reviews.append(movie_review)
print(all_movies_reviews)    








