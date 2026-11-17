"""CinéRadar web app (owner: Benoît).

Run: streamlit run app/streamlit_app.py  (the API must be running)
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.title("CinéRadar")
st.caption("Hidden gems for cinephiles")

user_id = st.number_input("User ID", min_value=1, value=1, step=1)
alpha = st.slider("Mainstream ↔ Cinephile", 0.0, 1.0, 0.5)

if st.button("Recommend"):
    response = requests.get(f"{API_URL}/recommend", params={"user_id": user_id, "alpha": alpha}, timeout=10)
    for rec in response.json()["recommendations"]:
        col1, col2 = st.columns([4, 1])
        # TODO: show the title and poster from TMDB instead of the id
        col1.write(f"Film {rec['movie_id']} (score {rec['score']})")
        if col2.button("Like", key=f"like-{rec['movie_id']}"):
            requests.post(f"{API_URL}/log_interaction",
                          json={"user_id": user_id, "movie_id": rec["movie_id"], "type": "like"}, timeout=10)
