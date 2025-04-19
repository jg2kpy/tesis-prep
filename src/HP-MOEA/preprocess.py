import pickle
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle

count = 0

def main(dataset_path, output_path = './data', top_usuarios = 10000, new_comer_filter = 5, random_seed = 69):
    # Leer el dataset
    print("\nLeyendo el dataset desde:", dataset_path)
    df = pd.read_csv(dataset_path)

    # Ajustar los IDs de usuario para que comiencen desde 0
    print("Ajustando los IDs de usuario para que comiencen desde 0")
    df.userId = df.userId - 1

    # Crear un mapeo para los IDs de películas
    print("Creando un mapeo para los IDs de películas")
    unique_movie_ids = set(df.movieId.values)
    movie2idx = {}
    count = 0
    for movie_id in unique_movie_ids:
        movie2idx[movie_id] = count
        count += 1

    df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

    # Eliminar la columna 'timestamp'
    print("Eliminando la columna 'timestamp'")
    df = df.drop(columns='timestamp')

    print("\nTamaño del dataframe original:", len(df))


    # Contar la cantidad de ratings por usuario
    print("Contando la cantidad de ratings por usuario")
    user_ids_count = Counter(df.userId)

    # Seleccionar los usuarios más comunes
    print(f"Seleccionando los {top_usuarios} usuarios películas más comunes")
    user_ids = [u for u, c in user_ids_count.most_common(top_usuarios)]

    # Filtrar el dataframe para que solo contenga los usuarios seleccionados
    print("Filtrando el dataframe para que solo contenga los usuarios seleccionados")
    df_small = df[df.userId.isin(user_ids)].copy()

    # Crear nuevos mapeos para los IDs de usuario
    print("Creando nuevos mapeos para los IDs de usuario")
    new_user_id_map = {}
    i = 0
    for old in user_ids:
        new_user_id_map[old] = i
        i += 1

    df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId], axis=1)

    # Obtener los IDs de las películas que aparecen más de 5 veces y menos de 5 veces
    movie_ids_count = Counter(df_small.movie_idx)
    movies_more_than_5 = [movie_id for movie_id, count in movie_ids_count.items() if count > new_comer_filter]
    new_comers = [movie_id for movie_id, count in movie_ids_count.items() if count <= new_comer_filter]

    print(f"Películas que aparecen más de {new_comer_filter} veces: {len(movies_more_than_5)}")
    print(f"Películas que aparecen menos o igual a {new_comer_filter} veces (new_comers): {len(new_comers)}")

    print("Creando nuevos mapeos para los IDs de peliculas")
    new_movie_id_map = {}
    i = 0
    for old in movies_more_than_5:
        new_movie_id_map[old] = i
        i += 1

    for old in new_comers:
        new_movie_id_map[old] = i
        i += 1

    df_small.loc[:, 'movie_idx'] = df_small.apply(lambda row: new_movie_id_map[row.movie_idx], axis=1)

    print("ID máximo de usuario:", df_small.userId.max())
    print("ID máximo de película:", df_small.movie_idx.max())

    print("Tamaño del dataframe reducido:", len(df_small))

    # Barajar el dataframe
    print("\nBarajando el dataframe")
    df_shuffle = shuffle(df_small, random_state=random_seed)
    df_shuffle_len = len(df_shuffle)

    user2movie = {}
    movie2user = {}
    usermovie2rating = {}
    movie2profit = {}
    print("Llamando a: update_user2movie_and_movie2user")
    def update_user2movie_and_movie2user(row):
        global count
        count += 1
        if count % 10000 == 0:
            print(f'Procesado: {((float(count) / df_shuffle_len) * 100):.2f}%')

        i = int(row.userId)
        j = int(row.movie_idx)

        if i not in user2movie:
            user2movie[i] = [j]
        else:
            user2movie[i].append(j)

        if j not in movie2user:
            movie2user[j] = [i]
        else:
            movie2user[j].append(i)

        usermovie2rating[(i, j)] = row.rating
    df_shuffle.apply(update_user2movie_and_movie2user, axis=1)

    def calcular_profit(ratings):
        if not ratings:
            return 0
        positive_ratings = sum(1 for rating in ratings if rating >= 3)
        return 1 + (9 * (positive_ratings / len(ratings)))

    # Calcular el "profit" para cada película basado en sus calificaciones
    print("Calculando el profit para cada película")
    for movie_id, users in movie2user.items():
        ratings = [usermovie2rating[(user, movie_id)] for user in users]
        profit = calcular_profit(ratings) if ratings else 0
        movie2profit[movie_id] = profit

    # Guardar los diccionarios en archivos
    print("Guardando las estructuras de datos en archivos")
    with open(f'{output_path}/user2movie.pickle', 'wb') as f:
        pickle.dump(user2movie, f)
        print(f"user2movie guardado en {output_path}/user2movie.pickle")

    with open(f'{output_path}/movie2user.pickle', 'wb') as f:
        pickle.dump(movie2user, f)
        print(f"movie2user guardado en {output_path}/movie2user.pickle")

    with open(f'{output_path}/usermovie2rating.pickle', 'wb') as f:
        pickle.dump(usermovie2rating, f)
        print(f"usermovie2rating guardado en {output_path}/usermovie2rating.pickle")

    with open(f'{output_path}/movie2profit.pickle', 'wb') as f:
        pickle.dump(movie2profit, f)
        print(f"movie2profit guardado en {output_path}/movie2profit.pickle")

    with open(f'{output_path}/new_comers.pickle', 'wb') as f:
        pickle.dump(new_comers, f)
        print(f"new_comers guardado en {output_path}/new_comers.pickle")

    with open(f'{output_path}/new_user_id_map.pickle', 'wb') as f:
        pickle.dump(new_user_id_map, f)
        print(f"new_user_id_map guardado en {output_path}/new_user_id_map.pickle")

    with open(f'{output_path}/new_movie_id_map.pickle', 'wb') as f:
        pickle.dump(new_movie_id_map, f)
        print(f"new_movie_id_map guardado en {output_path}/new_movie_id_map.pickle")

if __name__ == "__main__":
    main()
