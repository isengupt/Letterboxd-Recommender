from keras.utils import get_file
from itertools import chain
import tensorflow as tf
from collections import Counter, OrderedDict
import pickle
import json
import os
import numpy as np
import random
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
import tensorflowjs as tfjs
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15
random.seed(100)


class Embeddings:
    """ A keras model for letterboxd recommendations
    (10 most popular lists with movie in it were used to 
    generate "links" between movies with the 10 most linked
    movies to x movie being used as links). These links were then
    used for embeddings """ 
    def __init__(self, path, file_name):

        """ Get json data file from github """
        self.path = path
        self.file_name = file_name
        self.movies = []
        self.get_data()
        self.find_unique_movies()
        self.data_preparation()

    def get_data(self):

        def check_for_duplicates(movieArr):

            non_duplicate_arr = []
            for movie in movieArr:

                if movie in non_duplicate_arr:
                    continue
                else :
                    non_duplicate_arr.append(movie)

            return non_duplicate_arr

        def deleteDuplicatesByName(moviearr):
                movieNames = []
                movieComp = []

                for movie in moviearr:
                    if movie[0] in movieNames:
                        continue
                    else:
                        movieNames.append(movie[0])
                        movieComp.append(movie)
  
                return movieComp



        x = get_file(self.file_name, self.path)

        with open(x, 'r') as infile:

            self.movies = [json.loads(line) for line in infile]

        self.movies = [movie for movie in self.movies if "LetterboxdMovie" not in movie[0]]
        print(f'{len(self.movies)} found in {self.file_name} dataset')

        self.movies = check_for_duplicates(self.movies)
        self.movies = deleteDuplicatesByName(self.movies)

        print(f'{len(self.movies)} movies after duplicate deletion')

    def check_movie_info(self, index):

        """ Check info a selected index in the dataset """ 
        print(self.movies[index][0], self.movies[index][1], self.movies[index][2])

    def find_unique_movies(self):

        """ Find number of unique movies in dataset """
        self.movielinks = list(chain(*[movie[2] for movie in self.movies]))
        print(f'{len(set(self.movielinks))} unique movies found in the dataset')

    
    def data_preparation(self):

        """ Map movies to index numbers and index numbers to movies 
        and save indexes to pkl file
        """ 

        def count_items(l):
            """Count instances of each movie link in the dataset"""
    
            
            counts = Counter(l)
    
           
            counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)
            counts = OrderedDict(counts)
    
            return counts

        def save_obj(obj, name):
            with open(str(name) + ".pkl", "wb+") as f:
                pickle.dump(obj, f)

        self.movie_index = {movie[0]: idx for idx, movie in enumerate(self.movies)}
        self.index_movie = {idx: movie for movie, idx in self.movie_index.items()}

        save_obj(self.movie_index, "movie_index")
        save_obj(self.index_movie, "index_movie") 

        self.movielinks_other_movies = [movie for movie in self.movielinks if movie in self.movie_index.keys()]
        print(f"{len(set(self.movielinks_other_movies))} unique movie links to other movies")


        self.unique_movielinks = list(chain(*[list(set(movie[2])) for movie in self.movies]))
        self.movie_counts = count_items(self.unique_movielinks)
        print("Ten most linked movies")
        print(list(self.movie_counts.items())[:10])

        self.movielinks = [movielink for movielink in self.unique_movielinks]
        print(f"There are {len(set(self.movielinks))} unique wikilinks.")
        
        self.movielink_counts = count_items(self.movielinks)
        print(list(self.movielink_counts.items())[:10])

        self.links = [t[0] for t in self.movielink_counts.items() if t[1] >= 1]
        #filter items with low counts
        print(len(self.links))

        self.unique_movielink_movies =  list(chain(*[list(set(link for link in movie[2] if link in self.movie_index.keys())) for movie in self.movies]))

        self.movielink_movie_counts = count_items(self.unique_movielink_movies)
        print(list(self.movielink_movie_counts.items())[:10])   

        #pair links to indexes now
        self.link_index = {link: idx for idx, link in enumerate(self.links)}
        self.index_link = {idx: link for link, idx in self.link_index.items()}

        save_obj(self.index_link, "index_link")
        save_obj(self.link_index, "link_index")

        #construct pairs of links and movie title
        self.pairs = []
        for movie in self.movies:
            self.pairs.extend((self.movie_index[movie[0]], self.link_index[link]) for link in movie[2] if link in self.links)
        
        print(len(self.pairs), len(self.links), len(self.movies))
        self.pairs_set = set(self.pairs)

        x = Counter(self.pairs)

        print(sorted(x.items(), key = lambda x : x[1], reverse=True)[:10])

    def generate_batch(self, pairs, n_positive = 50, negative_ratio = 1.0, classification=True):

            """ Function to generate sample batches for data training """

            self.batch_size = n_positive * (1 + negative_ratio)
            self.batch = np.zeros((self.batch_size, 3))

            if classification:
                neg_label = 0
            else: 
                neg_label = -1

            while True:

                """ Choose random positive examples for training """
                for idx, (movie_id, link_id) in enumerate(random.sample(self.pairs, n_positive)):
                    self.batch[idx, :] = (movie_id, link_id, 1)

                idx += 1
                
                while idx < self.batch_size:
            
                    
                    random_movie = random.randrange(len(self.movies))
                    random_link = random.randrange(len(self.links))
            
                   
                    if (random_movie, random_link) not in self.pairs_set:
                
                        
                        self.batch[idx, :] = (random_movie, random_link, neg_label)
                        idx += 1
                
        
                np.random.shuffle(self.batch)
                yield {'movie': self.batch[:, 0], 'link': self.batch[:, 1]}, self.batch[:, 2]    
    
    def test_batches(self):
        x, y = next(self.generate_batch(n_positive=2, negative_ratio=2))

        for label, b_idx, l_idx in zip(y, x['movie'], x['link']):
            print(f'Movie: {self.index_movie[b_idx]:30} Link: {self.index_link[l_idx]:40} Label: {label}')

    def movie_embedding_model(self, embedding_size = 50, classification = True):
        """Model to embed movies and movie links using the functional API.
         """
    
        # Convert moves and links to 1 dimensional vectors
        movie = Input(name = 'movie', shape = [1])
        link = Input(name = 'link', shape = [1])
    
        # Embedding the movies in shape (None, 1, 50)
        movie_embedding = Embedding(name = 'movie_embedding',
                               input_dim = len(self.movie_index),
                               output_dim = embedding_size)(movie)
    
        # Embedding the links in shape (None, 1, 50))
        link_embedding = Embedding(name = 'link_embedding',
                               input_dim = len(self.link_index),
                               output_dim = embedding_size)(link)
    
        # Merge the layers with a dot product along the second axis with shape (None, 1, 1)
        merged = Dot(name = 'dot_product', normalize = True, axes = 2)([movie_embedding, link_embedding])
    
        # Reshape to be a single number with shape (None, 1)
        merged = Reshape(target_shape = [1])(merged)
    
        # Add extra layer for classification
        if classification:
            merged = Dense(1, activation = 'sigmoid')(merged)
            self.model = Model(inputs = [movie, link], outputs = merged)
            self.model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
       
        else:
            self.model = Model(inputs = [movie, link], outputs = merged)
            self.model.compile(optimizer = 'Adam', loss = 'mse')
    
        return self.model


    def find_similar(self, name, weights, index_name = 'movie', n= 10, least = False, return_dist = False, plot = False):
        if index_name == 'movie':
            index = self.movie_index
            rindex = self.index_movie
        elif index_name == 'page':
            index = self.link_index
            rindex = self.index_link

        try:
            """Search for movie in the index """ 
            self.dists = np.dot(weights, weights[index[name]])
        except KeyError:
            print(f'Movie with name {name} is not in the index.')
            return
        
        sorted_dists = np.argsort(self.dists)

            
    
    
        if least:
        
            closest = sorted_dists[:n]
         
            print(f'{index_name.capitalize()}s furthest from {name}.\n')
        
    
        else:
        
            closest = sorted_dists[-n:]
        
        
            if return_dist:
                return dists, closest
        
        
            print(f'{index_name.capitalize()}s closest to {name}.\n')
        
    
        if return_dist:
            return dists, closest
    
    
    
        max_width = max([len(rindex[c]) for c in closest])
    
    
        for c in reversed(closest):
            print(f'{index_name.capitalize()}: {rindex[c]:{max_width + 2}} Similarity: {self.dists[c]:.{2}}')

    def extract_weights(self, name, model):
        """Extract weights from a neural network model"""
    
        # Extract weights
        weight_layer = model.get_layer(name)
        weights = weight_layer.get_weights()[0]
    
        # Normalize
        weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
        return weights

    def get_weights(self, n_positive):
        self.model_small = self.movie_embedding_model(50, classification=True)
        self.gen = self.generate_batch(n_positive,negative_ratio=2, classification=True)

        self.h = self.model_small.fit_generator(self.gen, epochs = 15, steps_per_epoch = len(self.pairs) // n_positive, verbose = 0)
        export_path = 'linear_model/1/'
        tf.saved_model.save(self.model_small, os.path.join('./content/',export_path))
        path = "./model/"
        tfjs.converters.save_keras_model(self.model_small, path)
        self.movie_weights_class = self.extract_weights('movie_embedding', self.model_small)
        print(self.movie_weights_class.shape)



if __name__ == '__main__':
    model = Embeddings('https://raw.githubusercontent.com/isengupt/letterboxdRecommender/master/data/sampleMovies.ndjson', 'sampleMovies.ndjson')
    model_insta = model.movie_embedding_model()
    model_insta.summary()

    model.get_weights(n_positive=132)
    model.find_similar('Parasite', model.movie_weights_class, n = 5)
    model.find_similar('Hansel & Gretel: Witch Hunters', model.movie_weights_class, n = 5)
