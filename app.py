from flask import Flask, render_template, request
import main  # This is if you separate out your recommendation logic

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend(movie_name, df, sim_matrix):
    if movie_name not in df['title'].values:
        return ["Movie not found."]
    index = df[df['title'] == movie_name].index[0]
    distances = sorted(list(enumerate(sim_matrix[index])), key=lambda x: x[1], reverse=True)
    recommendations = [df.iloc[i[0]].title for i in distances[1:6]]
    return recommendations

if __name__ == '__main__':
    app.run(debug=True)
