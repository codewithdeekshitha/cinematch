import streamlit as st
import pickle
import requests

# -----------------------------
# TMDB API KEY
# -----------------------------
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")


# -----------------------------
# GET POSTER
# -----------------------------
def get_poster(movie_name):

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": API_KEY,
        "query": movie_name
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if len(data["results"]) > 0:

            poster_path = data["results"][0]["poster_path"]

            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

    except:
        return None

    return None


# -----------------------------
# LOAD MODEL
# -----------------------------
movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))


# -----------------------------
# PAGE
# -----------------------------
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 CineMatch")

st.subheader("Your Personal Movie Recommendation System 🍿")

st.write(
    "Choose a movie you like and CineMatch will recommend "
    "5 similar movies."
)


# -----------------------------
# SELECT MOVIE
# -----------------------------
selected_movie = st.selectbox(
    "🔍 Choose a movie you like:",
    movies["title"].values
)


# -----------------------------
# RECOMMEND
# -----------------------------
if st.button("🎯 Recommend Movies"):

    index = movies[
        movies["title"] == selected_movie
    ].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]


    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------
    st.subheader("🍿 Movies You May Like")

    cols = st.columns(5)

    for col, movie in zip(cols, movie_list):

        movie_name = movies.iloc[movie[0]]["title"]

        poster = get_poster(movie_name)

        with col:

            if poster:
                st.image(
                    poster,
                    use_container_width=True
                )
            else:
                st.info("Poster unavailable")

            st.write("🎬 **" + movie_name + "**")


    # -----------------------------
    # FEEDBACK
    # -----------------------------
    st.divider()

    st.subheader("⭐ Rate Your Recommendations")

    rating = st.slider(
        "How useful were these recommendations?",
        1,
        5,
        3
    )

    feedback = st.text_area(
        "💬 Tell us what you think:"
    )

    if st.button("📩 Submit Feedback"):

        st.success("Thank you for your feedback! ❤️")

        st.write("⭐ Your rating:", rating)

        if feedback:
            st.write("💬 Your feedback:", feedback)

        st.balloons()