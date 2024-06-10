import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Expanded sample data (user-item interactions)
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    'movie_id': [1, 2, 3, 2, 3, 1, 2, 3, 4, 1, 2, 4, 5, 2, 3, 5],
    'rating': [5, 4, 3, 4, 2, 3, 5, 4, 2, 4, 3, 5, 4, 5, 4, 3]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Create the user-item matrix
user_item_matrix = df.pivot(index='user_id', columns='movie_id', values='rating')

# Fill missing values with 0
user_item_matrix = user_item_matrix.fillna(0)

# Normalize ratings by subtracting mean movie rating
movie_item_matrix_normalized = user_item_matrix - user_item_matrix.mean(axis=0, skipna=True)

# Handle any potential NaN values that might have been introduced during normalization
movie_item_matrix_normalized = movie_item_matrix_normalized.fillna(0)

# Calculate cosine similarity between movies
item_similarity = cosine_similarity(movie_item_matrix_normalized.T)

# Convert the similarity matrix to a DataFrame for easier handling
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def get_item_based_recommendations(user_id, item_similarity_df, user_item_matrix, k=5):
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]

    # Get the movies the user has already rated
    rated_movies = user_ratings[user_ratings > 0].index

    # Calculate the weighted sum of movie similarities for each movie
    recommendations = pd.Series(dtype=float)
    for movie_id in rated_movies:
        similarity_scores = item_similarity_df[movie_id]
        user_rating = user_ratings[movie_id]
        recommendations = recommendations.add(similarity_scores * user_rating, fill_value=0)

    # Exclude movies the user has already rated
    recommendations = recommendations.drop(rated_movies)

    # Get the top k movie recommendations
    top_recommendations = recommendations.sort_values(ascending=False).head(k)

    return top_recommendations

# Example: Get top recommendations for user 1
user_id = 1
recommendations_user_1 = get_item_based_recommendations(user_id, item_similarity_df, user_item_matrix)
print(f"Top Recommendations for User {user_id}:")
print(recommendations_user_1)

# Example: Get top recommendations for user 4
user_id = 4
recommendations_user_4 = get_item_based_recommendations(user_id, item_similarity_df, user_item_matrix)
print(f"Top Recommendations for User {user_id}:")
print(recommendations_user_4)

# Example: Get top recommendations for user 5
user_id = 5
recommendations_user_5 = get_item_based_recommendations(user_id, item_similarity_df, user_item_matrix)
print(f"Top Recommendations for User {user_id}:")
print(recommendations_user_5)
