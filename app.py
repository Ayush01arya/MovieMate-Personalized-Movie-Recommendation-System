from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import ast

app = Flask(__name__)
CORS(app)

# Load the saved models
movies_list = pickle.load(open('movie_list.pkl', 'rb'))
import bz2

# Load compressed pickle file
with bz2.BZ2File('similarity_compressed.pkl', 'rb') as f_in:
    similarity = pickle.load(f_in)

# Load original dataset for additional movie details
movies_df = pd.read_csv('DataSets/tmdb_5000_movies.csv')


def get_movie_details(title):
    movie = movies_df[movies_df['title'] == title].iloc[0]
    return {
        'title': movie['title'],
        'overview': movie['overview'],
        'genres': [i['name'] for i in ast.literal_eval(movie['genres'])],
        'vote_average': round(float(movie['vote_average']), 1),
        'release_date': movie['release_date']
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/movies', methods=['GET'])
def get_movies():
    return jsonify({
        'movies': movies_list['title'].tolist()
    })


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        movie_name = data.get('movie')

        if movie_name not in movies_list['title'].values:
            return jsonify({
                'error': 'Movie not found'
            }), 404

        # Get selected movie details
        selected_movie = get_movie_details(movie_name)

        # Get recommendations
        index = movies_list[movies_list['title'] == movie_name].index[0]
        distances = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)

        recommendations = []
        for i in distances[1:6]:
            movie_title = movies_list.iloc[i[0]]['title']
            movie_info = get_movie_details(movie_title)
            movie_info['similarity_score'] = round(i[1] * 100, 2)
            recommendations.append(movie_info)

        return jsonify({
            'selected_movie': selected_movie,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)