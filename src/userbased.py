import os
import pickle
import numpy as np
from sortedcontainers import SortedList
import time

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
print("Numero total de itéms (peliculas): ", M)


#Calculate the weights
K = 25
limit = 5
neighbors = []
averages = []
deviations = []
start_time = time.time()

print('\nCalculando los pesos (weights)...')
print('Con la siguente configuración: ')
print('K: ', K, ' (cantidad maxima de vecinos que almacenaremos por cada usuario)')
print('limit: ', limit, ' (minima cantidad de itéms en comun que deben tener dos usuarios para calcular la correlación)\n')
for i in range(N):

    # Obtiene la lista de itéms que el usuario califico
    movies_i = user2movie[i]
    movies_i_set = set(movies_i)

    # Crea un diccionario itém:calificación de todas los itéms que el usuario califico
    ratings_i = { movie:usermovie2rating[(i, movie)] for movie in movies_i}
    # Calculo del promedio de las calificaciones del usuario
    avg_i = np.mean(list(ratings_i.values()))
    # Calcula la desviación que el usuario dio a cada calificación y lo almacena en un diccionario itém:desviación
    dev_i_dict = { movie:(rating - avg_i) for movie, rating in ratings_i.items()}
    # Obtiene la lista de desviaciones y calcula su cuadrado y lo almacena
    dev_i_values = np.array(list(dev_i_dict.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    # Almacena el promedio de las calificaciones del usuario y la cada unas de las desviaciones de sus califaciones
    averages.append(avg_i)
    deviations.append(dev_i_dict)

    # Aca hace el calculo del peso con otros usuarios y almacenamos solo los K usuarios con mayor peso
    sl = SortedList()
    for j in range(N): # Como podemos ver aca hacemos otro ciclo hasta N, por lo cual a partir de este momento el algoritmo es O(N ^ 2)
        if j != i: # No nos interesa calcular el peso con uno mismo
            # Obtenemos los itéms que el usuario j evaluo
            movies_j = user2movie[j]
            movies_j_set = set(movies_j)
            # Obtenemos una lista comun de itéms entre el usuario i y el usuario j
            common_movies = (movies_i_set & movies_j_set)
            if len(common_movies) > limit: # Solo nos interesa calcular el peso cuando la cantidad de itéms en comun es mayor a la variale limit
                # Hacemos el mismo procedimento del calculo de promedios y desviacions de arriba pero con el usuario j
                ratings_j = { movie:usermovie2rating[(j, movie)] for movie in movies_j}
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = { movie:(rating - avg_j) for movie, rating in ratings_j.items() }
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                # Y aca es donde calculamos el peso entre el usuario i y j mediante la correlacion de Pearson
                numerator = sum(dev_i_dict[m]*dev_j[m] for m in common_movies)
                w_ij = numerator / (sigma_i * sigma_j)


                # Almacenamos el peso calculado y el usuario en la lista ordenada
                sl.add((-w_ij, j))
                if len(sl) > K: # Si la lista supera la variable K, entonces eliminamos el que tenga menor peso
                    del sl[-1]

    # Almacenamos la lista ordenada como la lista de vecinos por cada usuario
    neighbors.append(sl)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    elapsed_time = time.time() - start_time
    percentage_completed = round((i / N, 4) * 100, 2)
    print(f"[{current_time}] Porcentaje completado: {percentage_completed}%, Tiempo transcurrido: {elapsed_time:.2f} segundos")

def predict(i, m):
    numerator = 0
    denominator = 0
    # Recorremos la lista de vecinos del usuario i, buscando si alguno de los vecinos califico el itém m
    for neg_w, j in neighbors[i]:
        try:
            # En numerador almacenamos la multiplicacion del peso entre el usuario i y j por la desviación del usuario j con el itém m
            numerator += -neg_w * deviations[j][m]
            # En denominador guardamos la suma de los pesos
            denominator += abs(neg_w)
        except KeyError:
            pass

    if denominator == 0: # Si el denominador es 0, solo nos queda usar el promedio de las califaciones del usuario i y j
        prediction = averages[i]
    else:
        # Finalmente la predicción seria, el promedio de calificaciones del usuario + la desviación que podria tener este usuario para este califcación en base a las calificaciones de otros usuarios similares
        prediction = averages[i] + (numerator / denominator)
    # Normalizamos la predición
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction

print('\nAhora empezamos a hacer predicciones')
train_predictions = []
train_targets = []
for (i, m), target in usermovie2rating.items():
    prediction = predict(i, m)

    # Guardamos la predición y el resultado que trendria que haber dado el sistema
    train_predictions.append(prediction)
    train_targets.append(target)

test_prediction = []
test_targets = []
for (i, m), target in usermovie2rating_test.items():
    prediction = predict(i, m)

    test_prediction.append(prediction)
    test_targets.append(target)

# Calculamos el error cuadrado medio
def mse(p ,t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t) ** 2)

print('\nError cuadrado medio comparando con los datos de entrenamiento:', mse(train_predictions, train_targets))
print('Error cuadrado medio comparando con los datos de prueba:', mse(test_prediction, test_targets))
