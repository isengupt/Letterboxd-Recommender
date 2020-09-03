import requests 
import csv
from bs4 import BeautifulSoup
import json

initialURL = "/film/parasite-2019/lists/by/popular/"
baseURL = "https://letterboxd.com"
listURL = "lists/by/popular/"
endingURL = "by/rating/"

class ThreadScraper:

    """ A class to scrape letterboxd movie info and linked 
    top lists with movie in it using
    multithreading (used for machine learning recommender system
    ) """

    def __init__(self, thread_num=4, save_path="movies.ndjson", limit):
        self.thread_num = thread_num
        self.limit = limit
        self.save_path = save_path
        self.movies_arr = []
        self.open_outfile
        self.get_movie_names

    def open_outfile(self):
        self.outfile = open(self.save_path, 'w')

    def get_movie_names(self, path="https://letterboxd.com/films/ajax/popular/page/1", base_path = "https://letterboxd.com/films/ajax/"):
        
        """ Scrape movies from ajax path until limit is reached """
        
        index = requests.get(path)
        soup = BeautifulSoup(index.text, 'html.parser')
        movie_title = soup.select('a[class="frame"]')
        for title in movie_title:
            self.outfile.write(json.dumps(title['href']) + "\n")
            self.movies_arr.append(title['href'])

        next_link = soup.select('div[class="paginate-nextprev"] > a[class="next"]')
        next_link_abridged = next_link[0]['href'].replace('/films/','')

        test_url = base_path + next_link_abridged

        if (len(self.movies_arr) > self.limit): 
            return self.movies_arr
        else:
            if (next_link):
                return get_movie_names(path = test_url)

    
    def thread_scrape_info(self, outfile_path, in_path):

        def get_movie_info(movie_url):

            """ Get movie info from letterboxd """

            movieInfo = {}
            index = requests.get(base_url + str(movie_url)) 
            soup = BeautifulSoup(index.text, 'html.parser')
            movieTitle = soup.select('section[id="featured-film-header"] > h1')
            if (movieTitle[0].text):     
                movieInfo['movieTitle'] = movieTitle[0].text
            movieYear = soup.select('small[class="number"] > a')
            if (movieYear[0].text):      
                movieInfo['movieYear'] = movieYear[0].text
            movieDescription = soup.select('div[class="truncate"] > p')
            if (movieDescription):
                movieInfo['movieDescription'] = movieDescription[0].text
            movieImage = soup.select('img[width="230"]')
            if (movieImage[0]['src']):
                movieInfo['movieImage'] = movieImage[0]['src']

            return movieInfo

        def listdictScraper(coreURL, listURLarr, datarr):
            """ Get data from movies in most popular list with the chosen movie""" 
            for listURL in listURLarr:
                index = requests.get(baseURL + listURL + endingURL)
                soup = BeautifulSoup(index.text, 'html.parser')
                for iteritem in soup.select('li > div'):
                    if (iteritem.has_attr('data-film-slug')):
                        if (iteritem['data-film-slug']) == coreURL:
                            continue
                    else:
                        if iteritem['data-film-slug'] in datarr:
                            datarr[iteritem['data-film-slug']] += 1
                        else: 
                            datarr[iteritem['data-film-slug']] = 1
        
    
            sorted_data = {k: v for k, v in sorted(
                datarr.items(), key=lambda item: item[1])}
            toptenLinks = list((sorted_data).items())[-10:]
            topLinksAppend = [x[0] for x in toptenLinks]
    
    
            return topLinksAppend

        def movielistScraper(movieURL, dataarr):
            index = ""
            if "lists/by/popular/page/" in movieURL:
                index = requests.get(baseURL + movieURL)
            else: 
                index = requests.get(baseURL + movieURL + listURL)
            soup = BeautifulSoup(index.text, 'html.parser')
            movieTitle = soup.select('a[class="list-link"]')
            for title in movieTitle:
                
                dataarr.append(title['href'])

            nextLink = soup.select('div[class="paginate-nextprev"] > a[class="next"]')
    
            nextLinkURL = nextLink[0]['href']
            if (len(dataarr) > 12):
        
                return dataarr

            else:
                if (nextLinkURL):
           
                    return movielistScraper(nextLinkURL, dataarr)
        
        def infoCompScraper(movieURL):
            
            movieInfo = movieinfoScraper(movieURL)
            movieLists = movielistScraper(movieURL, dataarr= [])
   
            topLinks = listdictScraper(movieURL, movieLists, datarr= {})
   
            movieComp = {}
            movieComp['movieInfo'] = movieInfo
            movieComp['topLinks'] = topLinks
    
            info_file.write(json.dumps(movieComp) + "\n")

            return movieComp

        def getMovieData():
            movieCollect = []
            with open(in_path, 'r') as fin:
                movieCollect = [json.loads(l) for l in fin]
            return movieCollect

        saved_links = getMovieData()

        start = timer()

        threadpool = Threadpool(processes=12)

        info_file =  open(outfile_path, 'w')   

        results = threadpool.map(infoCompScraper, saved_links)

        end = timer()

        print(f'Found {len(results)} books in {round(end - start)} seconds.')

if __name__ == '__main__':
    scraper = ThreadScraper(limit=10000)