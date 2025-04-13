from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load data
df = pd.read_pickle('books.pkl')
cosine_similarities_word2vec = cosine_similarity(list(df['word_embeddings']))

# Normalize strings
def normalize(text):
    return str(text).lower().strip()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    
    # Validate request data
    if not data or 'method' not in data or 'query' not in data:
        return jsonify({'error': 'Missing required fields: method or query'}), 400

    method = data['method'].lower()
    query = data['query']

    # Validate method
    if method not in ['title', 'genre', 'author', 'similar']:
        return jsonify({'error': 'Invalid recommendation method. Must be one of: title, genre, author, similar.'}), 400

    if method == 'title':
        return recommend_by_title(query)
    elif method == 'genre':
        return recommend_by_genre(query)
    elif method == 'author':
        return recommend_by_author(query)
    elif method == 'similar':
        return recommend_by_similarity(query)

def recommend_by_title(title):
    try:
        normalized_title = normalize(title)
        idx = df[df['title'].str.lower() == normalized_title].index[0]
    except IndexError:
        return jsonify({'error': 'Book not found'}), 404

    sim_scores = list(enumerate(cosine_similarities_word2vec[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    book_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[book_indices][['title', 'authors', 'categories']].to_dict(orient='records')

    return jsonify({'method': 'title', 'recommendations': recommendations})

def recommend_by_genre(genre):
    genre = normalize(genre)  # Normalize input genre
    genre_books = df[df['categories'].str.contains(genre, case=False, na=False)]
    
    if genre_books.empty:
        return jsonify({'error': 'No books found in this genre'}), 404
    
    recommendations = genre_books[['title', 'authors', 'categories']].head(10).to_dict(orient='records')
    return jsonify({'method': 'genre', 'recommendations': recommendations})

def recommend_by_author(author):
    author = normalize(author)  # Normalize input author
    author_books = df[df['authors'].str.contains(author, case=False, na=False)]
    
    if author_books.empty:
        return jsonify({'error': 'No books found by this author'}), 404
    
    recommendations = author_books[['title', 'authors', 'categories']].head(10).to_dict(orient='records')
    return jsonify({'method': 'author', 'recommendations': recommendations})

def recommend_by_similarity(title):
    try:
        normalized_title = normalize(title)
        idx = df[df['title'].str.lower() == normalized_title].index[0]
    except IndexError:
        return jsonify({'error': 'Book not found'}), 404

    sim_scores = list(enumerate(cosine_similarities_word2vec[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    book_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[book_indices][['title', 'authors', 'categories']].to_dict(orient='records')
    
    return jsonify({'method': 'similar', 'recommendations': recommendations})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

