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

    def add_movie(self, new_user):
        self.users.append(new_user)

    def get_all_id_users(self):
        return [user.id_user for user in self.users]

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
                return {'movies': user.movies, 'ratings': user.ratings}
        return None


class User():
    def __init__(self, id_user=None, movies=None, ratings=None, fromJson=None):
        if fromJson:
            self.id_user = fromJson['id_user']
            self.movies = fromJson['movies']
            self.ratings = fromJson['ratings']
        else:
            self.id_user = id_user
            self.movies = movies
            self.ratings = ratings

    def to_json(self):
        return {
            'id_user': self.id_user,
            'movies': self.movies,
            'ratings': self.ratings
        }
