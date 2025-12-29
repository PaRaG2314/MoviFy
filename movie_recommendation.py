# ----------------------------------------
# MoviFy - Movie Recommendation System
# With User Accounts + SQLite + Ollama DeepSeek Chatbot
# ----------------------------------------

import os
import time
import random
import sqlite3
import pandas as pd
import difflib
import requests
from functools import lru_cache
from flask import (
    Flask, request, render_template, jsonify,
    session, redirect, url_for
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY") or "movify_secret_key_123"
TMDB_KEY = os.getenv("TMDB_API_KEY")

# Ollama config (local DeepSeek)
# Make sure you have:
#   ollama run deepseek-r1:1.5b
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

print("Using Ollama model:", OLLAMA_MODEL)

app = Flask(__name__)
app.secret_key = SECRET_KEY

# -----------------------------
# SQLite User Accounts
# -----------------------------
DB_PATH = "users.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def create_user(username, password):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password))
        )
        conn.commit()
        conn.close()
        return True, None
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def authenticate(username, password):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    conn.close()
    if user and check_password_hash(user["password_hash"], password):
        return user
    return None


init_user_db()

# -----------------------------
# Load Movie CSV Files
# -----------------------------
PREFERRED_MOVIES_PATH = "/mnt/data/movies.csv"
PREFERRED_RATINGS_PATH = "/mnt/data/ratings.csv"


def load_csv(path1, fallback):
    if os.path.exists(path1):
        return pd.read_csv(path1, encoding="latin-1")
    if os.path.exists(fallback):
        return pd.read_csv(fallback, encoding="latin-1")
    raise FileNotFoundError(f"Missing both CSV files: {path1} & {fallback}")


movies_df = load_csv(PREFERRED_MOVIES_PATH, "movies.csv").head(1000)

try:
    ratings_df = load_csv(PREFERRED_RATINGS_PATH, "ratings.csv").head(50000)
except:
    ratings_df = pd.DataFrame(columns=["userId", "movieId", "rating"])

# -----------------------------
# Clean Movie Data
# -----------------------------
movies_df["clean_title"] = movies_df["title"].astype(str).str.replace(
    r"\s\(\d{4}\)$", "", regex=True
)

if "genres" not in movies_df.columns:
    movies_df["genres"] = ""

movies_df["genres"] = movies_df["genres"].fillna("").astype(str).str.replace("|", " ")

text_cols = []
for col in ["genres", "overview", "keywords", "cast", "director"]:
    if col in movies_df.columns:
        movies_df[col] = movies_df[col].fillna("").astype(str)
        text_cols.append(col)

if not text_cols:
    text_cols = ["clean_title"]

movies_df["soup"] = movies_df[text_cols].agg(" ".join, axis=1)

# -----------------------------
# TF-IDF Similarity
# -----------------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["soup"])
content_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(
    movies_df.index,
    index=movies_df["clean_title"].str.lower()
).drop_duplicates()

# -----------------------------
# Hybrid Rating System
# -----------------------------
if not ratings_df.empty and "movieId" in ratings_df.columns:
    avg_ratings = ratings_df.groupby("movieId")["rating"].mean().reset_index()
    avg_ratings.columns = ["movieId", "avg_rating"]
    movies_df = movies_df.merge(avg_ratings, on="movieId", how="left")
    movies_df["avg_rating"] = movies_df["avg_rating"].fillna(0)

    mn, mx = movies_df["avg_rating"].min(), movies_df["avg_rating"].max()
    movies_df["rating_norm"] = (movies_df["avg_rating"] - mn) / (mx - mn) if mx > mn else 0
else:
    movies_df["rating_norm"] = 0

# -----------------------------
# Poster Cache
# -----------------------------
POSTER_DIR = "static/posters"


def build_poster_cache():
    posters = {}
    if not os.path.isdir(POSTER_DIR):
        return posters

    files = set(os.listdir(POSTER_DIR))

    for _, row in movies_df.iterrows():
        t = row["clean_title"].lower()
        slug = t.replace(" ", "_")

        for ext in [".jpg", ".png", ".jpeg", ".webp"]:
            name = slug + ext
            if name in files:
                posters[t] = name
                break

    return posters


poster_cache = build_poster_cache()


def get_local_poster(title):
    return poster_cache.get(title.lower())

# -----------------------------
# Movie Popup Cache (FAST)
# -----------------------------
movie_popup_cache = {}

# -----------------------------
# Recommendation Page Cache
# -----------------------------
recommendation_cache = {}

# -----------------------------
# Streaming Providers (TMDb)
# -----------------------------
def get_streaming_providers(tmdb_id):
    """Fetch streaming availability for a movie in India (IN region)."""
    if not tmdb_id or not TMDB_KEY:
        return []

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/watch/providers"
    try:
        data = requests.get(
            url,
            params={"api_key": TMDB_KEY},
            timeout=8
        ).json()

        # Region = IN (India)
        region = data.get("results", {}).get("IN", {})
        providers = []

        # Check "flatrate", "rent", and "buy" sections
        for section in ["flatrate", "rent", "buy"]:
            if section in region:
                for p in region[section]:
                    providers.append({
                        "name": p.get("provider_name", "Unknown"),
                        "logo": p.get("logo_path")
                    })

        return providers

    except:
        return []

# -----------------------------
# External APIs (OMDb, YouTube)
# -----------------------------


@lru_cache(maxsize=2000)
def get_imdb(title):
    if not OMDB_API_KEY:
        return "N/A"
    try:
        data = requests.get(
            "http://www.omdbapi.com/",
            params={"apikey": OMDB_API_KEY, "t": title},
            timeout=5
        ).json()
        return data.get("imdbRating", "N/A")
    except:
        return "N/A"


@lru_cache(maxsize=2000)
def get_plot(title):
    if not OMDB_API_KEY:
        return "Plot unavailable"
    try:
        data = requests.get(
            "http://www.omdbapi.com/",
            params={"apikey": OMDB_API_KEY, "t": title, "plot": "short"},
            timeout=5
        ).json()
        return data.get("Plot", "Plot unavailable")
    except:
        return "Plot unavailable"


@lru_cache(maxsize=2000)
def get_trailers(title):
    if not YOUTUBE_API_KEY:
        return []
    try:
        data = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": f"{title} official trailer",
                "key": YOUTUBE_API_KEY,
                "type": "video",
                "maxResults": 3
            },
            timeout=5
        ).json()
        return [item["id"]["videoId"] for item in data.get("items", [])]
    except:
        return []

# -----------------------------
# Recommendation Function
# -----------------------------


def recommend(title):
    title = title.lower().strip()

    # ⚡ Return cached result if already computed
    if title in recommendation_cache:
        return recommendation_cache[title]

    all_titles = movies_df["clean_title"].str.lower().tolist()

    match = difflib.get_close_matches(title, all_titles, n=1, cutoff=0.3)
    if not match:
        return None, [], "Movie not found."

    idx = indices[match[0]]
    sim_scores = list(enumerate(content_sim[idx]))

    hybrid_scores = []
    for i, sim in sim_scores:
        hybrid_scores.append((i, 0.7 * sim + 0.3 * movies_df.iloc[i]["rating_norm"]))

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[1:6]

    main = movies_df.loc[idx, "clean_title"]

    main_movie = {
        "title": main,
        "poster": get_local_poster(main),
        "genres": movies_df.loc[idx, "genres"],
        "rating": get_imdb(main),
        "trailers": get_trailers(main)
    }

    # ⭐ Add streaming providers (using tmdb_id from movies_df, if available)
    tmdb_id = None
    if "tmdb_id" in movies_df.columns:
        tmdb_id = movies_df.loc[idx].get("tmdb_id", None)
    main_movie["providers"] = get_streaming_providers(tmdb_id)

    recs = []
    for i, _ in hybrid_scores:
        t = movies_df.loc[i, "clean_title"]
        recs.append({
            "title": t,
            "poster": get_local_poster(t),
            "genres": movies_df.loc[i, "genres"],
            "rating": get_imdb(t)
        })

    # ⚡ Cache full recommendation result
    recommendation_cache[title] = (main_movie, recs, None)

    return main_movie, recs, None


# -----------------------------
# Homepage (only movies with posters)
# -----------------------------
GENRES = ["Action", "Comedy", "Drama", "Horror",
          "Romance", "Thriller", "Animation", "Adventure"]

_cache = {"ts": 0, "data": None}


def get_homepage():
    now = time.time()
    if not _cache["data"] or now - _cache["ts"] > 60:

        data = {"trending": [], "genres": {}}

        df = movies_df[movies_df["clean_title"].apply(
            lambda t: get_local_poster(t) is not None
        )]

        try:
            pick = df.sample(min(20, len(df)))
            data["trending"] = [
                {"title": r["clean_title"], "poster": get_local_poster(r["clean_title"])}
                for _, r in pick.iterrows()
            ]
        except:
            pass

        for g in GENRES:
            subset = df[df["genres"].str.contains(g, case=False, na=False)]
            if not subset.empty:
                picks = subset.sample(min(18, len(subset)))
                data["genres"][g] = [
                    {"title": r["clean_title"], "poster": get_local_poster(r["clean_title"])}
                    for _, r in picks.iterrows()
                ]
            else:
                data["genres"][g] = []

        _cache["data"] = data
        _cache["ts"] = now

    return _cache["data"]

# -----------------------------
# Flask Routes (Auth + Pages)
# -----------------------------


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":
        u = request.form["username"].strip()
        p = request.form["password"]
        c = request.form["confirm"]

        if p != c:
            return render_template("register.html", error="Passwords do not match.")

        ok, msg = create_user(u, p)
        if ok:
            user = authenticate(u, p)
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))

        error = msg

    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":
        u = request.form["username"].strip()
        p = request.form["password"]
        user = authenticate(u, p)

        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))

        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def home():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/homepage_data")
def homepage_data():
    return jsonify(get_homepage())


@app.route("/recommend", methods=["POST"])
def recommend_movie():
    if not session.get("user_id"):
        return redirect(url_for("login"))

    query = request.form.get("movie", "")
    main, recs, msg = recommend(query)
    return render_template(
        "result.html",
        main_movie=main,
        recommendations=recs,
        suggestion=msg
    )


@app.route("/autocomplete")
def autocomplete():
    q = request.args.get("q", "").lower()
    matches = [t for t in movies_df["clean_title"].str.lower() if q in t][:8]
    return jsonify(matches)


@app.route("/movie_details/<path:title>")
def movie_details(title):
    t = title.strip()
    key = t.lower()

    # ⚡ instant return if cached
    if key in movie_popup_cache:
        return jsonify(movie_popup_cache[key])

    data = {
        "title": t,
        "poster": get_local_poster(t),
        "plot": get_plot(t),
        "rating": get_imdb(t),
        "trailers": get_trailers(t)
    }

    # store for next time
    movie_popup_cache[key] = data
    return jsonify(data)


# -----------------------------
# Ollama DeepSeek Chatbot Route
# -----------------------------


@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Please type a message."})

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"""You are a helpful movie assistant.
Give clear and concise answers.
Avoid unnecessary long explanations.

User: {message}
Assistant:""",
            "stream": False,
            "options": {
                "num_predict": 512,
                "temperature": 0.6
            }
        }

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=120
        )

        result = response.json()
        reply = result.get("response", "").strip()

        return jsonify({"reply": reply})

    except Exception:
        return jsonify({
            "reply": "AI service is taking too long or is temporarily unavailable. Please try again."
        })


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
