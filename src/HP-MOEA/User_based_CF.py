import os
import pickle
import numpy as np
from sortedcontainers import SortedList
import logger

neighbors = {}
averages = {}
deviations = {}

def main(data_path = './data/', K = 25, limit = 5, calcular_pesos = False, test = False):
    global neighbors, averages, deviations
    user2movie, movie2user, usermovie2rating, usermovie2rating_test = load_preprocessed_data(data_path)

    N = np.max(list(user2movie.keys())) + 1
    print("\nNumero total de usuarios: ", N)

    m1 = np.max(list(movie2user.keys())) + 1
    m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
    M = max(m1, m2)
    print("Numero total de ítems (peliculas): ", M)

    if calcular_pesos:
        neighbors, averages, deviations = calculate_weights(N, K, limit, user2movie, usermovie2rating)
        with open(f'{data_path}users_weights.json', 'wb') as f:
            pickle.dump((neighbors, averages, deviations), f)
    else:
        with open(f'{data_path}users_weights.json', 'rb') as f:
            neighbors, averages, deviations = pickle.load(f)

    if test:
        error_train, error_test = test_function(usermovie2rating, usermovie2rating_test)

    usermovie2predict_rating = make_predictions(N, M, user2movie)
    with open(f'{data_path}usermovie2predict_rating.json', 'wb') as f:
            pickle.dump(usermovie2predict_rating, f)

    print(f'\nTermino la ejecución del filtrado colaborativo basado en usuarios, los datos se guardaron en {data_path}usermovie2predict_rating.json')
    if test:
        print('Los resultados del testing fueron:')
        print('Error cuadrado medio comparando con los datos de entrenamiento:', error_train)
        print('Error cuadrado medio comparando con los datos de prueba:', error_test)

    return usermovie2predict_rating

def load_preprocessed_data(data_path):
        print('Cargamos los datos preprocesados...')
        user2movie = {}
        movie2user = {}
        usermovie2rating = {}
        usermovie2rating_test = {}

        if os.path.exists(f'{data_path}user2movie.json') and \
            os.path.exists(f'{data_path}movie2user.json') and \
            os.path.exists(f'{data_path}usermovie2rating.json') and \
            os.path.exists(f'{data_path}usermovie2rating_test.json'):

            with open(f'{data_path}user2movie.json', 'rb') as f:
                user2movie = pickle.load(f)

            with open(f'{data_path}movie2user.json', 'rb') as f:
                movie2user = pickle.load(f)

            with open(f'{data_path}usermovie2rating.json', 'rb') as f:
                usermovie2rating = pickle.load(f)

            with open(f'{data_path}usermovie2rating_test.json', 'rb') as f:
                usermovie2rating_test = pickle.load(f)

        print('\nLongitud de los datos preprocesados:')
        print("user2movie: ", len(user2movie))
        print("movie2user: ", len(movie2user))
        print("usermovie2rating: ", len(usermovie2rating))
        print("usermovie2rating_test: ", len(usermovie2rating_test))

        return user2movie, movie2user, usermovie2rating, usermovie2rating_test

def calculate_weights(N, K, limit, user2movie, usermovie2rating):
        neighbors = []
        averages = []
        deviations = []
        log = logger.logger_class('Calculo de pesos')

        print('\nCalculando los pesos (weights)...')
        print('Con la siguente configuración:')
        print('K: ', K, ' (cantidad maxima de vecinos que almacenaremos por cada usuario)')
        print('limit: ', limit, ' (minima cantidad de ítems en comun que deben tener dos usuarios para calcular la correlación)\n')
        for i in range(N):
            avg_i, dev_i_dict, sigma_i = calculate_user_stats(i, user2movie[i], usermovie2rating)
            averages.append(avg_i)
            deviations.append(dev_i_dict)

            sl = SortedList()
            for j in range(N):
                if j != i:
                    common_movies = (set(user2movie[i]) & set(user2movie[j]))
                    if len(common_movies) > limit:
                        avg_j, dev_j, sigma_j = calculate_user_stats(j, user2movie[j], usermovie2rating)
                        if sigma_i > 0 and sigma_j > 0:  # Verificar que sigma_i y sigma_j no sean cero
                            numerator = sum(dev_i_dict[m] * dev_j[m] for m in common_movies)
                            w_ij = numerator / (sigma_i * sigma_j)
                            sl.add((-w_ij, j))
                            if len(sl) > K:
                                del sl[-1]
            neighbors.append(sl)

            log.percentage(i, N)

        print('Termino el calculo de los pesos')

        return neighbors, averages, deviations

def calculate_user_stats(user, movies, usermovie2rating):
    ratings = {movie: usermovie2rating[(user, movie)] for movie in movies}
    avg_rating = np.mean(list(ratings.values()))
    deviations = {movie: (rating - avg_rating) for movie, rating in ratings.items()}
    deviation_values = np.array(list(deviations.values()))
    sigma = np.sqrt(deviation_values.dot(deviation_values))
    return avg_rating, deviations, sigma

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

def make_predictions(N, M, user2movie):
    print('\nIniciamos las predicciones con User Based Collaborative Filtering...')
    total_movies = set(range(M))
    usermovie2predict_rating = {}

    log = logger.logger_class('Prediccion')
    for i in range(N):
        user2movie_i = set(user2movie[i])
        user2predictmovies = list(total_movies - user2movie_i)

        for movie in user2predictmovies:
            usermovie2predict_rating[(i, movie)] = predict(i, movie)

        log.percentage(i, N)

    print('Termino la predicción')

    return usermovie2predict_rating

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

def mse(p ,t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t) ** 2)

if __name__ == "__main__":
    main()
