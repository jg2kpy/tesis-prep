import pickle
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

import os
if not os.path.exists('./preprocess/data/user2movie.json') or \
    not os.path.exists('./preprocess/data/movie2user.json') or \
    not os.path.exists('./preprocess/data/usermovie2rating.json') or \
    not os.path.exists('./preprocess/data/usermovie2rating_test.json'):
    import preprocess.preprocess2dict

with open('./preprocess/data/user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)

with open('./preprocess/data/movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)

with open('./preprocess/data/usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)

with open('./preprocess/data/usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

print(len(user2movie))
print(len(movie2user))
print(len(usermovie2rating))
print(len(usermovie2rating_test))


N = np.max(list(user2movie.keys())) + 1

m1 = np.max(list(movie2user.keys())) + 1
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print('N: ', N, 'M: ', M)


if N > 1000:
    print("N =", N, "are you sure you want to continue?")
    print("Comment out these lines so...")
    exit()



#Calculate the weights
K = 25
limit = 5
neighbors = []
averages = []
deviations = []

for i in range(N):

    movies_i = user2movie[i]
    movies_i_set = set(movies_i)

    ratings_i = { movie:usermovie2rating[(i, movie)] for movie in movies_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = { movie:(rating - avg_i) for movie, rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    averages.append(avg_i)
    deviations.append(dev_i)

    sl = SortedList()
    for j in range(N):
        if j != i:
            movies_j = user2movie[j]
            movies_j_set = set(movies_j)
            common_movies = (movies_i_set & movies_j_set)
            if len(common_movies) > limit:
                ratings_j = { movie:usermovie2rating[(j, movie)] for movie in movies_j}
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = { movie:(rating - avg_j) for movie, rating in ratings_j.items() }
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
                w_ij = numerator / (sigma_i * sigma_j)


                sl.add((-w_ij, j))
                if len(sl) > K:
                    del sl[-1]

    neighbors.append(sl)

    if i % 1 == 0:
        print(i)

def predict(i, m):
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        try:
            numerator += -neg_w * deviations[j][m]
            denominator += abs(neg_w)
        except KeyError:
            pass

    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction