import json

class Movies_info():
    def __init__(self, fromJson=None):
        self.movies_info = []
        if isinstance(fromJson, str):
            try:
                movies = json.loads(fromJson)
                self.movies_info = [
                    Movie_info(fromJson=movie) for movie in movies
                ]
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided")
        elif fromJson is not None:
            raise TypeError("fromJson must be a string containing valid JSON or None")

    def to_json(self):
        return json.dumps([movie.to_json() for movie in self.movies_info])

    def add_movie(self, new_movie_info):
        self.movies_info.append(new_movie_info)

    def get_len_movies(self):
        return len(self.movies_info)

    def get_all_id_movies(self):
        return [movie.id_movie for movie in self.movies_info]

    def get_profit_by_id_movie(self, id_movie):
        for movie in self.movies_info:
            if movie.id_movie == id_movie:
                return movie.profit
        return None

    def is_new_comer_by_id_movie(self, id_movie):
        for movie in self.movies_info:
            if movie.id_movie == id_movie:
                return movie.is_newcomer
        return None

    def get_all_newcomers(self):
        all_newcomers = []
        for movie in self.movies_info:
            if movie.is_newcomer:
                all_newcomers.append(movie)
        return all_newcomers


class Movie_info():
    def __init__(self, id_movie=None, profit=None, is_newcomer=None, fromJson=None):
        if fromJson:
            self.id_movie = fromJson['id_movie']
            self.profit = fromJson['profit']
            self.is_newcomer = fromJson['is_newcomer']
        else:
            self.id_movie = id_movie
            self.profit = profit
            self.is_newcomer = is_newcomer

    def to_json(self):
        return {
            'id_movie': self.id_movie,
            'profit': self.profit,
            'is_newcomer': self.is_newcomer
        }
