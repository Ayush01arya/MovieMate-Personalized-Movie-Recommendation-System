import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading and merging datasets
movi = pd.read_csv('DataSets/tmdb_5000_movies.csv')
cred = pd.read_csv('DataSets/tmdb_5000_credits.csv')
movi = movi.merge(cred, on='title')
movi = movi[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movi.dropna(inplace=True)

# Helper functions for data conversion
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert3(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def convert4(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

# Applying conversion functions
movi['genres'] = movi['genres'].apply(convert)
movi['keywords'] = movi['keywords'].apply(convert)
movi['cast'] = movi['cast'].apply(convert3)
movi['crew'] = movi['crew'].apply(convert4)

# Text preprocessing
def clean_text(text_list):
    return [i.replace(" ", "").lower() for i in text_list]

movi['overview'] = movi['overview'].apply(lambda x: x.split())
for col in ['overview', 'genres', 'keywords', 'cast', 'crew']:
    movi[col] = movi[col].apply(clean_text)

movi['tag'] = movi['overview'] + movi['genres'] + movi['cast'] + movi['crew'] + movi['keywords']
new_df = movi[['movie_id', 'title', 'tag']].copy()
new_df['tag'] = new_df['tag'].apply(lambda x: " ".join(x))

# Vectorization and similarity calculation
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tag']).toarray()
similarity = cosine_similarity(vector)

def recommend(movie, df, sim_matrix):
    if movie not in df['title'].values:
        return "Movie not found."
    index = df[df['title'] == movie].index[0]
    distances = sorted(list(enumerate(sim_matrix[index])), key=lambda x: x[1], reverse=True)
    recommended_movies = "\n".join(df.iloc[i[0]].title for i in distances[1:6])
    return recommended_movies

# User interaction
movie_name = input("Enter Movie Name :- ")
print(recommend(movie_name, new_df, similarity))

