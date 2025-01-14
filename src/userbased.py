import pickle
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

import os
if not os.path.exists('./preprocess/user2movie.json') or \
    not os.path.exists('./preprocess/movie2user.json') or \
    not os.path.exists('./preprocess/usermovie2rating.json') or \
    not os.path.exists('./preprocess/usermovie2rating_test.json'):
    import preprocess.preprocess2dict

with open('./preprocess/user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)

with open('./preprocess/movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)

with open('./preprocess/usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)

with open('./preprocess/usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

print(len(user2movie))
print(len(movie2user))
print(len(usermovie2rating))
print(len(usermovie2rating_test))

