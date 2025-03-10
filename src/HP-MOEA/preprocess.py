import pickle
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle

count = 0

def main(dataset_path, output_path = './data/', top_usuarios = 100000, top_peliculas = 20000):
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

    # Contar la cantidad de ratings por usuario y por película
    print("Contando la cantidad de ratings por usuario y por película")
    user_ids_count = Counter(df.userId)
    movie_ids_count = Counter(df.movie_idx)

    # Seleccionar los usuarios y películas más comunes
    print(f"Seleccionando los {top_usuarios} usuarios y {top_peliculas} películas más comunes")
    user_ids = [u for u, c in user_ids_count.most_common(top_usuarios)]
    movie_ids = [u for u, c in movie_ids_count.most_common(top_peliculas)]

    # Filtrar el dataframe para que solo contenga los usuarios y películas seleccionados
    print("Filtrando el dataframe para que solo contenga los usuarios y películas seleccionados")
    df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()

    # Crear nuevos mapeos para los IDs de usuario y película
    print("Creando nuevos mapeos para los IDs de usuario y película")
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
    print("ID máximo de usuario:", df_small.userId.max())
    print("ID máximo de película:", df_small.movie_idx.max())

    print("Tamaño del dataframe reducido:", len(df_small))

    # Barajar el dataframe
    print("\nBarajando el dataframe")
    df = shuffle(df)
    cutoff = int(0.8 * len(df))
    df_train = df.iloc[:cutoff]
    df_test = df.iloc[cutoff:]

    user2movie = {}
    movie2user = {}
    usermovie2rating = {}
    print("Llamando a: update_user2movie_and_movie2user")
    def update_user2movie_and_movie2user(row):
        global count
        count += 1
        if count % 1000000 == 0:
            print("Procesado: %.3f" % (float(count) / cutoff))

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
    df_train.apply(update_user2movie_and_movie2user, axis=1)

    usermovie2rating_test = {}
    print("Llamando a: update_usermovie2rating_test")
    count = 0
    def udpate_usermovie2rating_test(row):
        global count
        count += 1
        if count % 1000000 == 0:
            print("Procesado: %.3f" % (float(count) / len(df_test)))

        i = int(row.userId)
        j = int(row.movie_idx)
        usermovie2rating_test[(i, j)] = row.rating
    df_test.apply(udpate_usermovie2rating_test, axis=1)

    # Guardar los diccionarios en archivos
    print("Guardando los diccionarios en archivos")
    with open(f'{output_path}user2movie.json', 'wb') as f:
        pickle.dump(user2movie, f)
        print(f"user2movie guardado en {output_path}user2movie.json")

    with open(f'{output_path}movie2user.json', 'wb') as f:
        pickle.dump(movie2user, f)
        print(f"movie2user guardado en {output_path}movie2user.json")

    with open(f'{output_path}usermovie2rating.json', 'wb') as f:
        pickle.dump(usermovie2rating, f)
        print(f"usermovie2rating guardado en {output_path}usermovie2rating.json")

    with open(f'{output_path}usermovie2rating_test.json', 'wb') as f:
        pickle.dump(usermovie2rating_test, f)
        print(f"usermovie2rating_test guardado en {output_path}usermovie2rating_test.json")

if __name__ == "__main__":
    main()
