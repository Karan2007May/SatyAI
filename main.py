import streamlit as st
from deep_translator import GoogleTranslator
import requests
import pandas as pd
import random

# -------------------------------
# 🌍 SatyAI - An AI for Truth & Awareness
# -------------------------------

st.set_page_config(page_title="SatyAI - Truth & Awareness AI", layout="wide")

st.title("🧠 SatyAI – AI-Powered Truth & Awareness Platform")
st.write("Empowering citizens through verified facts, climate awareness, and multilingual education.")

# Sidebar Navigation
tabs = ["📰 Fake News Detector", "🌦️ Climate Awareness", "🤖 Ask SatyAI (Chatbot)"]
choice = st.sidebar.radio("Navigate", tabs)

# -------------------------------
# 📰 Fake News Detection (Placeholder Model)
# -------------------------------
if choice == tabs[0]:
    st.header("📰 Fake News Detector")
    st.write("Enter a news headline or short paragraph to check its authenticity.")
    
    user_input = st.text_area("Enter news text here:")
    
    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Placeholder AI logic (Random result for demo)
            result = random.choice(["Likely True ✅", "Suspicious ⚠️", "Fake ❌"])
            st.subheader("🧠 AI Analysis Result:")
            st.success(result)
            st.caption("*(Note: This is a demo version. The full model uses NLP and BERT for real detection.)*")

# -------------------------------
# 🌦️ Climate Awareness Section
# -------------------------------
elif choice == tabs[1]:
    st.header("🌦️ Climate & Environmental Awareness")
    st.write("Check real-time weather and air quality for your location.")
    
    city = st.text_input("Enter your city name:", "Bhopal")
    
    if st.button("Get Climate Data"):
        try:
            # Using Open-Meteo (free API)
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude=23.25&longitude=77.41&current_weather=true"
            data = requests.get(weather_url).json()
            
            temp = data["current_weather"]["temperature"]
            wind = data["current_weather"]["windspeed"]
            
            st.metric(label="🌡️ Temperature (°C)", value=temp)
            st.metric(label="💨 Wind Speed (km/h)", value=wind)
            
            st.success("✅ Data fetched successfully.")
            st.caption("*(Note: You can later integrate OpenWeatherMap or AQI APIs for richer data.)*")
        except Exception as e:
            st.error("Error fetching data. Please check your internet connection or city name.")

# -------------------------------
# 🤖 Chatbot Section (English–Hindi Translator + Awareness Chat)
# -------------------------------
elif choice == tabs[2]:
    st.header("🤖 Ask SatyAI – Your Awareness Assistant")
    st.write("Ask any question related to environment, awareness, or general knowledge (English or Hindi).")
    
    question = st.text_input("Ask here:")
    
    if st.button("Ask SatyAI"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            try:
                # Detect & Translate to English if needed
                translated_q = GoogleTranslator(source='auto', target='en').translate(question)
                
                # Simple rule-based responses (you can replace with LangChain or LLM later)
                if "climate" in translated_q.lower():
                    answer = "Climate change refers to long-term shifts in temperature and weather patterns, mainly due to human activities."
                elif "pollution" in translated_q.lower():
                    answer = "Pollution harms the planet by contaminating air, water, and soil. Planting trees and reducing plastic use helps reduce it."
                elif "recycle" in translated_q.lower():
                    answer = "Recycling saves resources and energy. Always separate dry and wet waste properly."
                else:
                    answer = "That’s a great question! SatyAI is still learning. Try asking about climate, pollution, or recycling."
                
                # Translate back to Hindi if question was in Hindi
                detected_lang = GoogleTranslator(source='auto', target='en').translate(question)
                if detected_lang != question:
                    translated_ans = GoogleTranslator(source='en', target='hi').translate(answer)
                    st.success(translated_ans)
                else:
                    st.success(answer)
            except Exception as e:
                st.error("⚠️ Translation or AI module error.")
                st.caption(str(e))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💡 *Developed by Karan Gattani | Powered by Python & Open Data | SatyAI © 2025*")
