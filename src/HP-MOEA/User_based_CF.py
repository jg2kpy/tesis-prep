import os
import pickle
import numpy as np
from sortedcontainers import SortedList
import time

def load_data(data_path):
    print('Cargamos los datos preprocesados...')
    if not os.path.exists('./preprocess/data/user2movie.json') or \
        not os.path.exists('./preprocess/data/movie2user.json') or \
        not os.path.exists('./preprocess/data/usermovie2rating.json') or \
        not os.path.exists('./preprocess/data/usermovie2rating_test.json'):

        with open('./preprocess/data/user2movie.json', 'rb') as f:
            user2movie = pickle.load(f)

        with open('./preprocess/data/movie2user.json', 'rb') as f:
            movie2user = pickle.load(f)

        with open('./preprocess/data/usermovie2rating.json', 'rb') as f:
            usermovie2rating = pickle.load(f)

        with open('./preprocess/data/usermovie2rating_test.json', 'rb') as f:
            usermovie2rating_test = pickle.load(f)

    print('\nLongitud de los datos preprocesados:')
    print("user2movie: ", len(user2movie))
    print("movie2user: ", len(movie2user))
    print("usermovie2rating: ", len(usermovie2rating))
    print("usermovie2rating_test: ", len(usermovie2rating_test))

def main(data_path, k, limit, test = False):
    print(data_path)

if __name__ == "__main__":
    main()
