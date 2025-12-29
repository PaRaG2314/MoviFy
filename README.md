# 🎬 MoviFy – AI-Powered Movie Recommendation System

MoviFy is an intelligent movie recommendation platform that combines:

- **Machine Learning-based recommendations**  
- **Flask backend with MongoDB**  
- **AI-powered chatbot using Ollama (DeepSeek R1)**  
- **OMDb + YouTube Trailer integration**  
- **Netflix-style UI with dynamic movie posters background**

This project provides an immersive experience for users looking to explore, search, and discover movies with a modern, cinematic interface.

---

## 🚀 Features

### 🔍 **Smart Movie Search**
- Autocomplete suggestions  
- Search by full or partial movie title  
- Clean and responsive search bar  

### 🎞️ **Movie Details Modal**
When clicking any movie card:
- Poster  
- IMDb rating  
- Plot summary  
- Trailer (YouTube iframe)  
- Smooth animation & blur background  

### 🤖 **MoviFy AI Chatbot**
Built using:
- **Ollama**
- **DeepSeek R1:1.5B model**

The chatbot can:
- Recommend movies  
- Explain plots  
- Suggest genres  
- Act as a movie guide  

### 🎬 **Dynamic Homepage**
- Scrambled/blurred movie posters as background  
- Trending movies  
- Genre sections  
- Netflix-inspired hover effects  

### 🔐 **User System**
- Register  
- Login  
- Logout  
- Session-based auth  

### 📁 **Backend Includes**
- Flask  
- MongoDB  
- Machine Learning similarity (TF-IDF / cosine similarity)  
- OMDb API  
- YouTube API for trailer retrieval  
- Fully structured endpoints  

---

## 🛠️ **Tech Stack**

### **Frontend**
- HTML5  
- CSS3 (custom Netflix-style UI)  
- JavaScript  

### **Backend**
- Python  
- Flask  
- MongoDB  
- Jinja2 templates  
- scikit-learn  
- requests / pandas  

### **AI**
- Ollama (local LLM server)  
- DeepSeek R1 1.5B model  
- Flask REST API  

### **APIs Used**
- OMDb API (movie details)  
- YouTube Data API (trailers)

---

## 📝 **Project Structure**

movie_recommender/
│── static/
│ └── posters/
│── templates/
│ ├── index.html
│ ├── login.html
│ ├── register.html
│ ├── result.html
│── download_posters.py
│── movie_recommendation.py
│── movies.csv
│── ratings.csv (ignored from Git)
│── users.db
│── .env (ignored)
│── .gitignore
│── README.md

