import numpy as np
import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD DATA
# ==========================================

movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")


# ==========================================
# 2. MERGE DATASETS
# ==========================================

movies = movies.merge(credits, on="title")


# ==========================================
# 3. SELECT IMPORTANT COLUMNS
# ==========================================

movies = movies[
    ["movie_id", "title", "overview",
     "genres", "keywords", "cast", "crew"]
]


# ==========================================
# 4. REMOVE MISSING VALUES
# ==========================================

movies.dropna(inplace=True)


# ==========================================
# 5. CONVERT GENRES & KEYWORDS
# ==========================================

def convert(obj):
    L = []

    for i in ast.literal_eval(obj):
        L.append(i["name"])

    return L


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)


# ==========================================
# 6. GET TOP 3 ACTORS
# ==========================================

def convert3(obj):
    L = []

    for i in ast.literal_eval(obj)[:3]:
        L.append(i["name"])

    return L


movies["cast"] = movies["cast"].apply(convert3)


# ==========================================
# 7. GET DIRECTOR
# ==========================================

def fetch_director(obj):
    L = []

    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])

    return L


movies["crew"] = movies["crew"].apply(fetch_director)


# ==========================================
# 8. CREATE TAGS
# ==========================================

movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x)) + " " +
    movies["cast"].apply(lambda x: " ".join(x)) + " " +
    movies["crew"].apply(lambda x: " ".join(x))
)


# ==========================================
# 9. CONVERT TAGS TO LOWERCASE
# ==========================================

movies["tags"] = movies["tags"].apply(lambda x: x.lower())


# ==========================================
# 10. CREATE FINAL DATAFRAME
# ==========================================

new_df = movies[["movie_id", "title", "tags"]]


# ==========================================
# 11. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = vectorizer.fit_transform(new_df["tags"]).toarray()

print("Vector shape:", vectors.shape)


# ==========================================
# 12. COSINE SIMILARITY
# ==========================================

similarity = cosine_similarity(vectors)

print("Similarity shape:", similarity.shape)


# ==========================================
# 13. RECOMMENDATION FUNCTION
# ==========================================

def recommend(movie):

    index = new_df[new_df["title"] == movie].index[0]

    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nRecommended Movies:")

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# ==========================================
# 14. TEST THE SYSTEM
# ==========================================

recommend("Avatar")

import pickle

pickle.dump(new_df, open("movies.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))