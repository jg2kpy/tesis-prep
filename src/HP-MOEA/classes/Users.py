import json

class Users():
    def __init__(self, fromJson=None):
        self.users = []
        if isinstance(fromJson, str):
            try:
                users = json.loads(fromJson)
                self.users = [
                    User(fromJson=movie) for movie in users
                ]
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided")
        elif fromJson is not None:
            raise TypeError("fromJson must be a string containing valid JSON or None")

    def to_json(self):
        return json.dumps([user.to_json() for user in self.users])

    def add_user(self, new_user):
        self.users.append(new_user)

    def get_user_by_id(self, id_user):
        for user in self.users:
            if user.id_user == id_user:
                return user
        return None

    def get_or_create_user_by_id(self, id_user):
        for user in self.users:
            if user.id_user == id_user:
                return user
        user = User(id_user)
        self.users.append(user)
        return user

    def get_all_id_users(self):
        return [user.id_user for user in self.users]

    def get_all_users(self):
        return self.users

    def get_len_users(self):
        return len(self.users)

    def get_movies_by_id_user(self, id_user):
        for user in self.users:
            if user.id_user == id_user:
                return user.movies
        return None

    def get_ratings_by_id_user(self, id_user):
        for user in self.users:
            if user.id_user == id_user:
                return user.ratings
        return None

    def get_movies_and_ratings_by_id_user(self, id_user):
        for user in self.users:
            if user.id_user == id_user:
                return [{'movie': movie, 'rating': rating} for movie, rating in zip(user.movies, user.ratings)]
        return None

    def get_ratings_by_id_movie(self, id_movie):
        ratings = []
        for user in self.users:
            for movie, rating in zip(user.movies, user.ratings):
                if movie == id_movie:
                    ratings.append(rating)
        return ratings

    def set_users(self, new_users):
        self.users = new_users

    def delete_movie(self, id_movie):
        count = 0
        for user in self.users:
            if user.delete_movie(id_movie):
                count+=1
                if user.get_len_movie() == 0:
                    self.users.remove(user)
        return count

class User():
    def __init__(self, id_user=None, fromJson=None):
        if fromJson:
            self.id_user = fromJson['id_user']
            self.movies = fromJson['movies']
            self.ratings = fromJson['ratings']
        else:
            self.id_user = id_user
            self.movies = []
            self.ratings = []

    def to_json(self):
        return {
            'id_user': self.id_user,
            'movies': self.movies,
            'ratings': self.ratings
        }

    def get_id(self):
        return self.id_user

    def add_movie_rating(self, new_id_movie, new_rating):
        self.movies.append(new_id_movie)
        self.ratings.append(new_rating)

    def get_movie_ids(self):
        return self.movies

    def get_ratings(self):
        return self.ratings

    def get_rating_by_movie_id(self, movie_id):
        position = self.movies.index(movie_id)
        return self.ratings[position]

    def get_movies_and_ratings(self):
        return list(zip(self.movies, self.ratings))

    def delete_movie(self, id_movie):
        if id_movie in self.movies:
            position = self.movies.index(id_movie)
            self.ratings.pop(position)
            self.movies.pop(position)
            return True
        return False

    def get_len_movie(self):
        return len(self.movies)
