import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("dataset.csv")

# Fill missing overviews
df['overview'] = df['overview'].fillna('')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie titles to indices
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommendation function
def recommend(title, cosine_sim=cosine_sim):
    if title not in indices:
        print("Movie not found in database.")
        return
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Example usage
user_input = input("Enter a movie title: ")
recommendations = recommend(user_input)
if recommendations is not None:
    print("\nRecommended Movies:")
    for movie in recommendations:
        print("-", movie)
