import os
import pickle
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

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


K = 10 # latent dimensionality
W = np.random.randn(N, K)
b = np.zeros(N)
U = np.random.randn(M, K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))

# prediction[i, j] = W[i].dot(U[j]) + b[i] + c.T[j] + mu

def get_loss(d):
    N = float(len(d))
    sse = 0
    for k, r in d.items():
        i, j = k
        p = W[i].dot(U[j]) + b[i] + c.T[j] + mu
        sse += (p - r) * (p - r)
    return sse / N


epochs = 25
reg = 20.
train_losses = []
test_losses = []
for epoch in range(epochs):
    print("epochs: ", epochs)
    epoch_start = datetime.now()

    t0 = datetime.now()
    for i in range(N):
        # for W
        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        # for b
        bi = 0
        for j in user2movie[i]:
            r = usermovie2rating[(i, j)]
            matrix += np.outer(U[j], U[j])
            vector += (r - b[i] - c[i] - mu) * U[j]
            bi += (r - W[i].dot(U[j]) - c[j] - mu)

        # set the updates
        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(user2movie[i]) + reg)

        if i % (N//10) == 0:
            print("i: ", i, "N: ", N)
    print("updated W and b:", datetime.now() - t0)

    t0 = datetime.now()
    for j in range(M):
        # for U
        matrix = np.eye(K) * reg
        vector = np.zeros(K)

        # for c
        cj = 0
        try:
            for i in movie2user[j]:
                r = usermovie2rating[(i, j)]
                matrix += np.outer(W[j], W[j])
                vector += (r - b[i] - c[i] - mu) * W[j]
                cj += (r - W[i].dot(U[j]) - c[j] - mu)

            W[i] = np.linalg.solve(matrix, vector)
            cj = cj / (len(user2movie[i]) + reg)

            if j % (M//10) == 0:
                print("j:", j, "M:", M)
        except KeyError:
            # possible not to have any ratings for a movie
            pass

        print("updated U and c: ", datetime.now() - t0)
        print("epoch duration", datetime.now() - epoch_start)

    t0 = datetime.now()
    train_losses.append(get_loss(usermovie2rating))

    test_losses.append(get_loss(usermovie2rating_test))
    print("calculate cost:", datetime.now() - t0)
    print("train loss:", datetime.now() - t0)
    print("test loss:", datetime.now() - t0)


print("train losses:", train_losses)
print("test losses:", test_losses)

plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()
