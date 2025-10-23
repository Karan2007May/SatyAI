# main.py
import streamlit as st
from deep_translator import GoogleTranslator
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
import sqlite3
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging

# ---------------------------
# Configuration & Constants
# ---------------------------
APP_TITLE = "SatyAI — Truth & Climate Awareness"
DB_PATH = "satyai_logs.db"
MODEL_PATH = "fake_news_model.joblib"
VECT_PATH = "faq_vectorizer.joblib"

# Minimal fallback dataset for demo training (very small)
FALLBACK_DATA = [
    {"text": "Govt confirms new relief funds for flood victims", "label": "real"},
    {"text": "Celebrity endorses miracle drug cures COVID instantly", "label": "fake"},
    {"text": "Local authorities issue heatwave advisory for next week", "label": "real"},
    {"text": "Drinking salt water prevents infection says viral post", "label": "fake"},
    {"text": "City council announces tree-planting drive in schools", "label": "real"},
    {"text": "Fake image circulating shows politician in hospital — edited", "label": "fake"}
]

# Basic FAQ corpus for chatbot (expand later)
FAQ_CORPUS = [
    ("What to do during a heatwave?",
     "Stay hydrated, avoid direct sun between 11 AM - 4 PM, use fans/AC if available, check on elders and children."),
    ("How to stay safe during floods?",
     "Move to higher ground, avoid contact with floodwater, follow official evacuation orders, don't drive through flooded roads."),
    ("How to spot fake news?",
     "Check source credibility, look for reputable outlets, read beyond the headline, check fact-checkers like PIB Fact Check."),
    ("What is climate change?",
     "Climate change refers to long-term shifts in temperatures and weather patterns, primarily due to human activities like burning fossil fuels."),
    ("How to check air quality?",
     "Access official AQI portals or local government sensors. If AQI is poor, avoid outdoor exercise and use masks if required.")
]

# Setup logging
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Utility functions
# ---------------------------

@st.cache_resource
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            module TEXT,
            input_text TEXT,
            result_text TEXT
        )
    ''')
    conn.commit()
    return conn

def log_event(conn, module, input_text, result_text):
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        c = conn.cursor()
        c.execute('INSERT INTO logs (timestamp, module, input_text, result_text) VALUES (?, ?, ?, ?)',
                  (ts, module, input_text[:200], result_text[:1000]))
        conn.commit()
    except Exception as e:
        logging.exception("Failed to log event: %s", e)

# Load or train a simple fake-news classifier
@st.cache_data(show_spinner=False)
def get_fake_news_pipeline(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        try:
            pipe = joblib.load(model_path)
            logging.info("Loaded saved fake-news model.")
            return pipe
        except Exception as e:
            logging.warning("Could not load model, will retrain fallback. %s", e)

    # Build fallback dataset DataFrame
    df = pd.DataFrame(FALLBACK_DATA)
    X = df['text']
    y = df['label'].map({"fake": 0, "real": 1})  # fake=0, real=1

    # Simple TF-IDF + Logistic Regression pipeline
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=1)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X, y)
    try:
        joblib.dump(pipe, model_path)
        logging.info("Trained and saved fallback fake-news model.")
    except Exception as e:
        logging.warning("Could not save model: %s", e)
    return pipe

# Chatbot retrieval vectorizer
@st.cache_data(show_spinner=False)
def get_faq_vectorizer(corpus=FAQ_CORPUS, vect_path=VECT_PATH):
    texts = [q.lower() + " " + a.lower() for (q,a) in corpus]
    if os.path.exists(vect_path):
        try:
            vec = joblib.load(vect_path)
            vectors = vec.transform(texts)
            logging.info("Loaded saved FAQ vectorizer.")
            return vec, vectors
        except Exception as e:
            logging.warning("Failed to load FAQ vectorizer, rebuilding. %s", e)

    vec = TfidfVectorizer(ngram_range=(1,2))
    vectors = vec.fit_transform(texts)
    try:
        joblib.dump(vec, vect_path)
    except Exception:
        logging.info("Could not persist FAQ vectorizer — continuing in-memory.")
    return vec, vectors

def retrieve_faq_answer(question, corpus=FAQ_CORPUS, vec=None, vectors=None):
    if vec is None or vectors is None:
        vec, vectors = get_faq_vectorizer()
    qry = question.lower()
    qv = vec.transform([qry])
    sims = cosine_similarity(qv, vectors)[0]
    idx = int(np.argmax(sims))
    # Return the answer part
    return corpus[idx][1], float(sims[idx])

# Safe translation wrapper
def translate_text(text, target='en'):
    try:
        if not text.strip():
            return text
        return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception as e:
        logging.warning("Translation failed: %s", e)
        return text

# ---------------------------
# Climate functions (Open-Meteo)
# ---------------------------

def get_lat_lon_for_city(city_name):
    # Use open API geocoding from open-meteo
    try:
        params = {"name": city_name, "count": 1, "format": "json"}
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=params, timeout=10)
        data = r.json()
        if data.get("results"):
            first = data["results"][0]
            return float(first["latitude"]), float(first["longitude"]), first.get("country","")
        return None
    except Exception as e:
        logging.warning("Geocoding error: %s", e)
        return None

@st.cache_data(ttl=300)
def fetch_weather_and_aqi(lat, lon):
    try:
        # Current weather
        params_weather = {"latitude": lat, "longitude": lon, "current_weather": True}
        w = requests.get("https://api.open-meteo.com/v1/forecast", params=params_weather, timeout=10).json()
        current = w.get("current_weather", {})
        # Air quality (Open-Meteo has a separate endpoint for air_quality)
        params_aq = {"latitude": lat, "longitude": lon, "hourly": "pm10,pm2_5", "timezone": "UTC"}
        aq = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=params_aq, timeout=10).json()
        return {"weather": current, "aq": aq.get("hourly", {})}
    except Exception as e:
        logging.exception("fetch_weather_and_aqi failed: %s", e)
        return {}

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🌎 " + APP_TITLE)
st.caption("Integrated Fake News Detection • Climate Awareness Map • Lightweight Chatbot • Local Logging")

# Initialize DB and models
conn = init_db()
fake_news_pipe = get_fake_news_pipeline()
faq_vec, faq_vectors = get_faq_vectorizer()

# Layout: sidebar navigation
menu = st.sidebar.selectbox("Go to", ["Fake News Detector", "Climate Map & Data", "Ask SatyAI (Chatbot)", "Logs & Analytics", "About"])

# ---------------------------
# Page: Fake News Detector
# ---------------------------
if menu == "Fake News Detector":
    st.header("📰 Fake News Detector")
    st.write("Paste a news headline, URL excerpt, or article text. Model: TF-IDF + LogisticRegression (stable & lightweight).")
    user_text = st.text_area("Enter text to analyze:", height=160)
    use_external = st.checkbox("I have a dataset (CSV) to fine-tune the model (optional)", value=False)
    if use_external:
        uploaded = st.file_uploader("Upload CSV with columns 'text' and 'label' (label: real/fake)", type=["csv"])
    else:
        uploaded = None

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Analyze"):
            if not user_text.strip():
                st.warning("Please enter text to analyze.")
            else:
                try:
                    # If user uploaded dataset, quick fine-tune (tiny)
                    if uploaded is not None:
                        try:
                            df_up = pd.read_csv(uploaded)
                            if 'text' in df_up.columns and 'label' in df_up.columns:
                                X = df_up['text'].astype(str)
                                y = df_up['label'].map({"fake":0,"real":1})
                                # retrain small model
                                pipe = Pipeline([
                                    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=1)),
                                    ('clf', LogisticRegression(max_iter=1000))
                                ])
                                pipe.fit(X, y)
                                joblib.dump(pipe, MODEL_PATH)
                                fake_news_pipe = pipe
                                st.success("Model fine-tuned on your small dataset (saved).")
                            else:
                                st.warning("CSV must have 'text' and 'label' columns.")
                        except Exception as e:
                            st.error("Failed to use uploaded dataset: " + str(e))

                    proba = fake_news_pipe.predict_proba([user_text])[0]
                    pred = fake_news_pipe.predict([user_text])[0]
                    label = "Real ✅" if pred == 1 else "Fake ❌"
                    confidence = round(max(proba)*100,2)
                    st.metric("Prediction", label, delta=f"{confidence}% confidence")
                    explanation = "Model is a lightweight TF-IDF + Logistic Regression classifier trained on small/demo data. Replace with a larger dataset for production."
                    st.info(explanation)
                    log_event(conn, "fake_news", user_text, f"{label} ({confidence}%)")
                except Exception as e:
                    st.error("Analysis failed: " + str(e))

    with col2:
        st.subheader("Quick Tips")
        st.markdown("""
        - Check the source: prefer verified outlets.
        - Read full articles, not just headlines.
        - Use multiple sources for confirmation.
        - Expand training dataset for better results (Kaggle datasets are helpful).
        """)

# ---------------------------
# Page: Climate Map & Data
# ---------------------------
elif menu == "Climate Map & Data":
    st.header("🌍 Climate Map & Data")
    st.write("Enter a city to view location, current weather and basic air-quality info (when available). Map is interactive.")

    city = st.text_input("City name (e.g., Bhopal):", value="Bhopal")
    colA, colB = st.columns([2,1])
    with colB:
        if st.button("Fetch Data"):
            st.session_state['fetch_time'] = time.time()

    if st.button("Show on Map") or ('fetch_time' in st.session_state):
        if not city.strip():
            st.warning("Enter a city name.")
        else:
            with st.spinner("Looking up city coordinates..."):
                loc = get_lat_lon_for_city(city)
            if not loc:
                st.error("Could not find the city. Try a different name.")
            else:
                lat, lon, country = loc
                st.success(f"Found: {city}, {country} ({lat:.3f}, {lon:.3f})")
                data = fetch_weather_and_aqi(lat, lon)
                weather = data.get("weather", {})
                aq = data.get("aq", {})

                # Show simple metrics
                c1, c2, c3 = st.columns(3)
                if weather:
                    c1.metric("Temperature (°C)", weather.get("temperature", "N/A"))
                    c2.metric("Wind Speed (m/s)", weather.get("windspeed", "N/A"))
                    c3.metric("Weather Code", weather.get("weathercode", "N/A"))
                else:
                    st.info("Weather data not available.")

                # AQ preview
                if aq and 'pm2_5' in aq:
                    try:
                        latest_pm25 = aq['pm2_5'][-1]
                        st.metric("PM2.5 (latest)", f"{latest_pm25:.2f} µg/m³")
                    except Exception:
                        st.info("AQ data fetched but couldn't parse latest values.")
                else:
                    st.info("No AQ data from server for this location.")

                # Folium Map
                map_center = [lat, lon]
                m = folium.Map(location=map_center, zoom_start=9)
                folium.Marker(location=map_center,
                              popup=f"{city}: Temp {weather.get('temperature','N/A')}°C",
                              tooltip=city).add_to(m)
                st_data = st_folium(m, width=900, height=500)

                # Log
                log_event(conn, "climate_lookup", city, f"lat={lat},lon={lon}")

# ---------------------------
# Page: Ask SatyAI (Chatbot)
# ---------------------------
elif menu == "Ask SatyAI (Chatbot)":
    st.header("🤖 Ask SatyAI — Lightweight Awareness Assistant")
    st.write("Ask questions about climate, safety, or identification of suspicious news. Supports English & Hindi translation.")
    user_q = st.text_input("Ask a question (English or Hindi):")
    if st.button("Get Answer"):
        if not user_q.strip():
            st.warning("Please type a question.")
        else:
            try:
                # Translate to English for processing
                q_en = translate_text(user_q, target='en')
                # Attempt retrieval from FAQ
                answer, score = retrieve_faq_answer(q_en, vec=faq_vec, vectors=faq_vectors)
                # If low similarity, fallback to short rule-based heuristics
                if score < 0.2:
                    ql = q_en.lower()
                    if "fake" in ql or "fake news" in ql or "hoax" in ql:
                        answer = "To check a news item, verify source, read multiple reliable outlets, and consult fact-checkers like PIB Fact Check."
                    elif "heat" in ql or "heatwave" in ql:
                        answer = "During heatwaves: hydrate, avoid midday sun, check vulnerable people, and follow local advisories."
                    elif "flood" in ql:
                        answer = "During floods: move to higher ground, avoid wading in floodwater, and stay tuned to official alerts."
                    else:
                        answer = "SatyAI is still learning. Try rephrasing your question or ask about climate, pollution, or fake news."
                # Translate answer back to user's language preference (detect by comparing)
                # If user_q likely in Hindi (non-ASCII heuristic), translate back to Hindi
                is_hindi = any("\u0900" <= ch <= "\u097F" for ch in user_q)
                if is_hindi:
                    answer_out = translate_text(answer, target='hi')
                else:
                    answer_out = answer
                st.success(answer_out)
                log_event(conn, "chatbot", user_q, answer_out)
            except Exception as e:
                st.error("Failed to answer: " + str(e))

    st.markdown("**Sample prompts:** What to do during a heatwave? | How to spot fake news?")

# ---------------------------
# Page: Logs & Analytics
# ---------------------------
elif menu == "Logs & Analytics":
    st.header("📊 Logs & Simple Analytics")
    st.write("Local logs of user interactions (stored in SQLite). Use this for basic report generation.")

    try:
        df_logs = pd.read_sql_query("SELECT * FROM logs ORDER BY id DESC LIMIT 200", conn)
        st.dataframe(df_logs)
        st.markdown("---")
        st.write("Summary by module:")
        if not df_logs.empty:
            counts = df_logs['module'].value_counts()
            st.bar_chart(counts)
        else:
            st.info("No logs yet — interact with modules to generate logs.")
    except Exception as e:
        st.error("Could not load logs: " + str(e))

# ---------------------------
# Page: About
# ---------------------------
else:
    st.header("About SatyAI")
    st.markdown("""
    **SatyAI** is a learning project by Rtr. Karan Gattani.
    It integrates:
    - Fake news detection (lightweight and trainable)
    - Climate data + interactive map (Open-Meteo)
    - A fast retrieval-based chatbot (FAQ + rules)
    - Local logging with SQLite
    """)
    st.markdown("**Next improvements you can add:**")
    st.markdown("- Replace fallback fake-news model with a large labeled dataset and retrain (Colab recommended).")
    st.markdown("- Integrate official AQI sources for richer air-quality info.")
    st.markdown("- Use a hosted LLM or LangChain for richer chatbot answers (careful with tokens).")
    st.markdown("- Add authentication and export for logs (CSV/PDF).")

# Footer
st.markdown("---")
st.caption("SatyAI — built with ❤️ by Karan Gattani. Keep improving & be responsible with AI.")
