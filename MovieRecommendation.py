import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset with features (genres and actors)
movies = pd.DataFrame({
    'Title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'Genres': ['Action', 'Drama', 'Comedy', 'Action'],
    'Actors': ['Actor X, Actor Y', 'Actor Y, Actor Z', 'Actor X, Actor Z', 'Actor Y, Actor Z']
})

# Combine text features into a single column for CountVectorizer
movies['Features'] = movies['Genres'] + ' ' + movies['Actors']

# Create CountVectorizer matrix
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(movies['Features'])

# Calculate cosine similarity between movies
cosine_sim = cosine_similarity(feature_matrix)

# Function to get movie recommendations
def get_movie_recommendations(movie_title, cosine_sim_matrix, data_frame):
    idx = data_frame.index[data_frame['Title'] == movie_title].tolist()[0]
    similar_scores = list(enumerate(cosine_sim_matrix[idx]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_movies_indices = [i[0] for i in similar_scores]
    return data_frame['Title'].iloc[similar_movies_indices[1:]]  # Exclude the input movie itself

# Test the recommendation system
movie_to_recommend = 'Movie A'
recommendations = get_movie_recommendations(movie_to_recommend, cosine_sim, movies)
print(f"Recommendations for '{movie_to_recommend}':")
print(recommendations)
