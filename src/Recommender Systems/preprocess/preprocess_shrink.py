import pandas as pd
from collections import Counter

#Leer el dataset
df = pd.read_csv('./data/edited_rating.csv')
print("original dataframe size", len(df))

N = df.userId.max() + 1
M = df.movie_idx.max() + 1

user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movie_idx)

n = 10000
m = 2000

user_ids = [u for u, c in user_ids_count.most_common(n)]
movie_ids = [u for u, c in movie_ids_count.most_common(n)]

df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()

new_user_id_map = {}
i = 0
for old in user_ids:
    new_user_id_map[old] = i
    i += 1

new_movie_id_map = {}
j = 0
for old in movie_ids:
    new_movie_id_map[old] = j
    j += 1

df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId], axis=1)
df_small.loc[:, 'movie_idx'] = df_small.apply(lambda row: new_movie_id_map[row.movie_idx], axis=1)
print("max user id: ", df_small.userId.max())
print("max movie id: ", df_small.movie_idx.max())


print("small dataframe size: ", len(df_small))
df_small.to_csv('./data/very_small_rating.csv')
