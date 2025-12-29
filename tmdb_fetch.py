import requests
import pandas as pd
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"
POSTER_DIR = "static/posters"
os.makedirs(POSTER_DIR, exist_ok=True)


def fetch(url, params={}):
    params["api_key"] = API_KEY
    for _ in range(5):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except:
            time.sleep(1)
    return None


def get_movie_details(movie_id):
    data = fetch(f"{BASE_URL}/movie/{movie_id}", {
        "append_to_response": "credits,keywords"
    })
    if not data:
        return None

    title = data.get("title", "").strip()
    year = (data.get("release_date") or "0000")[:4]

    genres = " ".join([g["name"] for g in data.get("genres", [])])
    overview = data.get("overview") or ""

    keywords = " ".join(
        [k["name"] for k in data.get("keywords", {}).get("keywords", [])]
    )

    cast_list = [c["name"] for c in data.get("credits", {}).get("cast", [])[:5]]
    cast = " ".join(cast_list)

    crew = data.get("credits", {}).get("crew", [])
    director = next((c["name"] for c in crew if c["job"] == "Director"), "")

    poster_path = data.get("poster_path")
    poster_file = None

    if poster_path:
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        poster_file = f"{title.lower().replace(' ', '_')}.jpg"

        try:
            img_data = requests.get(poster_url).content
            with open(os.path.join(POSTER_DIR, poster_file), "wb") as f:
                f.write(img_data)
        except:
            poster_file = None

    return {
    "title": title,
    "year": year,
    "genres": genres,
    "overview": overview,
    "keywords": keywords,
    "cast": cast,
    "director": director,
    "poster": poster_file,
    "tmdb_id": movie_id   # ⭐ REQUIRED
}




def fetch_all_movies(pages=200):
    movies = []

    print("Fetching movies from TMDb…")

    for page in tqdm(range(1, pages + 1)):
        data = fetch(f"{BASE_URL}/movie/popular", {"page": page})
        if not data or "results" not in data:
            continue

        for m in data["results"]:
            movie_id = m["id"]
            details = get_movie_details(movie_id)
            if details:
                movies.append(details)

        time.sleep(0.2)

    return movies


movies = fetch_all_movies(50)  # 200 pages ≈ 20,000 movies
df = pd.DataFrame(movies)
df.to_csv("movies.csv", index=False, encoding="utf-8")

print("DONE! movies.csv generated successfully.")
