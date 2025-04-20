import os
import json
import pickle
import numpy as np
from sortedcontainers import SortedList
import logger

from classes.Movies_info import Movies_info, Movie_info
from classes.Users import Users

neighbors = {}
averages = {}
deviations = {}

def main(data_path = './data', K = 25, limit = 5, calcular_pesos = True, test = False):
    global neighbors, averages, deviations
    users2movie_ratings, movies_info = load_preprocessed_data(data_path)

    eliminados = delete_all_newcomers(users2movie_ratings, movies_info)

    print(f'\nCantidad de usuarios: {users2movie_ratings.get_len_users()}')
    print(f'Cantidad de peliculas: {movies_info.get_len_movies() - eliminados}')

    if test:
        testing_mode(users2movie_ratings, movies_info, K, limit)

    if calcular_pesos:
        neighbors, averages, deviations = calculate_weights(users2movie_ratings, K, limit)
        with open(f'{data_path}/users_weights.pickle', 'wb') as f:
            pickle.dump((neighbors, averages, deviations), f)
    else:
        with open(f'{data_path}/users_weights.pickle', 'rb') as f:
            neighbors, averages, deviations = pickle.load(f)

    users2predict_movie_ratings = make_predictions(users2movie_ratings, movies_info)

    with open(f'{data_path}/users2predict_movie_ratings.json', 'w') as f:
            f.write(users2predict_movie_ratings.to_json())

    print(f'\nTermino la ejecución del filtrado colaborativo basado en usuarios, los datos se guardaron en {data_path}/usermovie2predict_rating.json')

    return users2predict_movie_ratings

def load_preprocessed_data(data_path):
        print('Cargamos los datos preprocesados...')
        users2movie_ratings_json = {}
        movies_info_json = {}

        if os.path.exists(f'{data_path}/users2movie_ratings.json') and \
            os.path.exists(f'{data_path}/movies_info.json'):

            with open(f'{data_path}/users2movie_ratings.json', 'r') as f:
                users2movie_ratings_json = f.read()

            with open(f'{data_path}/movies_info.json', 'r') as f:
                movies_info_json = f.read()

            print('\nLongitud de los datos preprocesados:')
            print("users2movie_ratings: ", len(users2movie_ratings_json))
            print("movies_info: ", len(movies_info_json))

            users2movie_ratings = Users(fromJson=users2movie_ratings_json)
            movies_info = Movies_info(fromJson=movies_info_json)

            return users2movie_ratings, movies_info

def calculate_weights(users2movie_ratings, K, limit):
        log = logger.logger_class('Calculo de pesos')

        print('\nCalculando los pesos (weights)...')
        print('Con la siguente configuración:')
        print('K: ', K, ' (cantidad maxima de vecinos que almacenaremos por cada usuario)')
        print('limit: ', limit, ' (minima cantidad de ítems en comun que deben tener dos usuarios para calcular la correlación)\n')

        users = users2movie_ratings.get_all_users()
        users_len = users2movie_ratings.get_len_users()
        neighbors = {}
        averages = {}
        deviations = {}
        for i, user_i in enumerate(users):
            user_i_id = user_i.get_id()
            dev_i_dict, sigma_i = calculate_user_stats(user_i, averages, deviations)

            sl = SortedList()
            for user_j in users:
                user_j_id = user_j.get_id()
                if user_i_id != user_j_id:
                    common_movies = (set(user_i.get_movie_ids()) & set(user_j.get_movie_ids()))
                    if len(common_movies) > limit:
                        dev_j, sigma_j = calculate_user_stats(user_j, averages, deviations)
                        if sigma_i > 0 and sigma_j > 0:  # Verificar que sigma_i y sigma_j no sean cero
                            numerator = sum(dev_i_dict[m] * dev_j[m] for m in common_movies)
                            w_ij = numerator / (sigma_i * sigma_j)
                            sl.add((-w_ij, user_j.get_id()))
                            if len(sl) > K:
                                del sl[-1]
            neighbors[user_i_id] = sl

            log.percentage(i, users_len)

        log.success('Termino el calculo de los pesos')

        return neighbors, averages, deviations

def calculate_user_stats(user, averages, deviations_array):
    user_id = user.get_id()
    if user_id in averages.keys():
        avg_rating = averages[user_id]
        deviations = deviations_array[user_id]
    else:
        movies_ratings = user.get_movies_and_ratings()
        avg_rating = np.mean([rating for _, rating in movies_ratings])
        deviations = {movie: (rating - avg_rating) for movie, rating in movies_ratings}
        averages[user_id] = avg_rating
        deviations_array[user_id] = deviations

    deviation_values = np.array(list(deviations.values()))
    sigma = np.sqrt(deviation_values.dot(deviation_values))
    return deviations, sigma

def predict(i, m):
    global neighbors, averages, deviations
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
        prediction = averages[i] + (numerator / denominator)

    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction

def make_predictions(users2movie_ratings, movies_info):
    print('\nIniciamos las predicciones con User Based Collaborative Filtering...')
    all_movies_ids = set(movies_info.get_all_id_movies())
    users2predict_movie_ratings = Users()

    log = logger.logger_class('PREDICCIÓN')
    users = users2movie_ratings.get_all_users()
    users_len = users2movie_ratings.get_len_users()
    for i, user in enumerate(users):
        user_id = user.get_id()
        movies_user_rated = set(user.get_movie_ids())
        movies_user_dont_rate = list(all_movies_ids - movies_user_rated)

        new_user = users2predict_movie_ratings.get_or_create_user_by_id(user_id)

        for movie in movies_user_dont_rate:
            predict_rating = predict(user_id, movie)
            new_user.add_movie_rating(movie, predict_rating)

        log.percentage(i, users_len)

    log.success('Termino la predicción')

    return users2predict_movie_ratings

def test_function(usermovie2rating, usermovie2rating_test):
    print('\nIniciando testing...')
    train_predictions = []
    train_targets = []
    log = logger.logger_class('Testing con train set')
    contador = 0
    for (i, m), target in usermovie2rating.items():
        contador+=1
        prediction = predict(i, m)

        train_predictions.append(prediction)
        train_targets.append(target)

        log.percentage(contador, len(usermovie2rating))

    test_prediction = []
    test_targets = []
    log = logger.logger_class('Testing con test set')
    contador = 0
    for (i, m), target in usermovie2rating_test.items():
        contador+=1
        prediction = predict(i, m)

        test_prediction.append(prediction)
        test_targets.append(target)
        log.percentage(contador, len(usermovie2rating_test))

    error_train = mse(train_predictions, train_targets)
    error_test = mse(test_prediction, test_targets)

    print('Termino el testing')
    #print('\nError cuadrado medio comparando con los datos de entrenamiento:', error_train)
    #print('Error cuadrado medio comparando con los datos de prueba:', error_test)
    #print('Continuando la ejecución en 10 segundos')
    #time.sleep(10)
    return error_train, error_test

def delete_all_newcomers(users2movie_ratings, movies_info):
    print('Eliminando newcomers...')
    log = logger.logger_class('DELETING_NEWCOMERS')
    newcomers = movies_info.get_all_newcomers()
    count = 0
    for i, newcomer in enumerate(newcomers):
        count += users2movie_ratings.delete_movie(newcomer.id_movie)
        log.percentage(i, len(newcomers))
    log.success(f'Se eliminaron {count} calificaciones por ser de newcomers')
    return len(newcomers)

def testing_mode():
    pass

def mse(p ,t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t) ** 2)

if __name__ == "__main__":
    main()
