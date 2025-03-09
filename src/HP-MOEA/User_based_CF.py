import os
import pickle
import numpy as np
from sortedcontainers import SortedList
import time

def load_preprocessed_data(data_path):
        print(f'{data_path}/user2movie.json')
        print('Cargamos los datos preprocesados...')
        user2movie = {}
        movie2user = {}
        usermovie2rating = {}
        usermovie2rating_test = {}

        if os.path.exists(f'{data_path}/user2movie.json') and \
            os.path.exists(f'{data_path}/movie2user.json') and \
            os.path.exists(f'{data_path}/usermovie2rating.json') and \
            os.path.exists(f'{data_path}/usermovie2rating_test.json'):

            with open(f'{data_path}/user2movie.json', 'rb') as f:
                user2movie = pickle.load(f)

            with open(f'{data_path}/movie2user.json', 'rb') as f:
                movie2user = pickle.load(f)

            with open(f'{data_path}/usermovie2rating.json', 'rb') as f:
                usermovie2rating = pickle.load(f)

            with open(f'{data_path}/usermovie2rating_test.json', 'rb') as f:
                usermovie2rating_test = pickle.load(f)

        print('\nLongitud de los datos preprocesados:')
        print("user2movie: ", len(user2movie))
        print("movie2user: ", len(movie2user))
        print("usermovie2rating: ", len(usermovie2rating))
        print("usermovie2rating_test: ", len(usermovie2rating_test))

        return user2movie, movie2user, usermovie2rating, usermovie2rating_test

def main(data_path = './data', k = 25, limit = 5, test = False):
    user2movie, movie2user, usermovie2rating, usermovie2rating_test = load_preprocessed_data(data_path)

if __name__ == "__main__":
    main()
