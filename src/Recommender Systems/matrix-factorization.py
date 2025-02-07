import os
import pickle

import numpy as np

print('Cargamos los datos preprocesados...')
if not os.path.exists('./preprocess/data/user2movie.json') or \
    not os.path.exists('./preprocess/data/movie2user.json') or \
    not os.path.exists('./preprocess/data/usermovie2rating.json') or \
    not os.path.exists('./preprocess/data/usermovie2rating_test.json'):
    print("Running preprocess...")
    import preprocess.preprocess
    import preprocess.preprocess_shrink
    import preprocess.preprocess2dict

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

N = np.max(list(user2movie.keys())) + 1
print("\nNumero total de usuarios: ", N)

m1 = np.max(list(movie2user.keys())) + 1
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2)
print("Numero total de ítems (peliculas): ", M)

