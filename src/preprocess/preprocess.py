import pandas as pd

#Leer el dataset
df = pd.read_csv('../movielens-20m-dataset/rating.csv')

#make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a maping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1

df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

df = df.drop(columns='timestamp')

df.to_csv('./data/edited_rating.csv')
