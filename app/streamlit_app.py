import os

import requests
import streamlit as st

API_URL = os.environ.get("CINERADAR_API", "http://localhost:8000")
MIN_LIKES = 5
BUTTONS = [("👍", "liked", "Liked it"), ("👎", "disliked", "Didn't like it"),
           ("👁️", "seen", "Already seen"), ("🚫", "not_interested", "Not interested")]

st.set_page_config(page_title="CinéRadar", page_icon="🎬", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=30, **kwargs)
    except requests.ConnectionError:
        st.error(f"The API is not reachable at {API_URL}. Start it with: uvicorn api.main:app")
        st.stop()
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(detail)
        return None
    return response.json()


def label(film: dict) -> str:
    year = f" ({film['year']})" if film.get("year") else ""
    return f"{film['title']}{year}"


def film_card(film: dict, user_id: int, key: str, conversation_id=None):
    with st.container(border=True):
        if film.get("poster_url"):
            st.image(film["poster_url"])
        badge = " 💎" if film.get("hidden_gem") else ""
        st.markdown(f"**{label(film)}**{badge}")
        st.caption(film.get("genres", ""))
        if film.get("because_title"):
            st.caption(f"Because you liked *{film['because_title']}*")
        if film.get("overview"):
            with st.expander("Summary"):
                st.write(film["overview"])
        columns = st.columns(len(BUTTONS))
        for column, (icon, kind, tip) in zip(columns, BUTTONS):
            if column.button(icon, key=f"{key}-{kind}-{film['movie_id']}", help=tip):
                api("POST", "/log_interaction", json={"user_id": user_id, "movie_id": film["movie_id"], "type": kind,
                                                      "source": "chat" if conversation_id else "click",
                                                      "conversation_id": conversation_id})
                st.toast(f"{tip}: {film['title']} saved to your profile")


def film_grid(films: list[dict], user_id: int, key: str, per_row: int = 5, conversation_id=None):
    for start in range(0, len(films), per_row):
        columns = st.columns(per_row)
        for column, film in zip(columns, films[start:start + per_row]):
            with column:
                film_card(film, user_id, key, conversation_id)


def sign_in():
    st.title("🎬 CinéRadar")
    st.subheader("Your film-loving friend who finds the hidden gems you'd never find alone")
    new, existing = st.columns(2)
    with new:
        st.markdown("#### New here?")
        pseudo = st.text_input("Pick a nickname")
        if st.button("Start", type="primary", disabled=not pseudo.strip()):
            user = api("POST", "/users", json={"pseudo": pseudo.strip()})
            if user:
                st.session_state.user_id = user["user_id"]
                st.rerun()
    with existing:
        st.markdown("#### Already have a profile?")
        user_id = st.number_input("Your user ID", min_value=1, step=1)
        if st.button("Continue"):
            if api("GET", f"/profile/{int(user_id)}"):
                st.session_state.user_id = int(user_id)
                st.rerun()


def questionnaire(user: dict):
    st.title(f"Welcome {user.get('pseudo') or ''} 👋")
    st.write("A few questions so I can get to know your taste. Every answer you give later will keep teaching me.")
    if "picked" not in st.session_state:
        st.session_state.picked = {}
    genres = api("GET", "/genres") or []

    st.markdown("#### 1. Genres")
    left, right = st.columns(2)
    genres_liked = left.multiselect("Genres you love", genres)
    genres_avoided = right.multiselect("Genres you avoid", [g for g in genres if g not in genres_liked])

    st.markdown(f"#### 2. Films you loved (at least {MIN_LIKES}, ideally 10)")
    params = {"limit": 120}
    if genres_liked:
        params["genre"] = genres_liked
    suggestions = api("GET", "/movies/popular", params=params) or []
    query = st.text_input("Search a film")
    if query.strip():
        suggestions = (api("GET", "/movies/search", params={"q": query, "limit": 20}) or []) + suggestions
    options = {f["movie_id"]: f for f in list(st.session_state.picked.values()) + suggestions}
    liked = st.multiselect("Pick the films you loved", list(options), format_func=lambda m: label(options[m]),
                           key="liked_select")
    for movie_id in liked:
        st.session_state.picked[movie_id] = options[movie_id]
    disliked = st.multiselect("And a few films you really didn't like (optional)",
                              [m for m in options if m not in liked], format_func=lambda m: label(options[m]))

    st.markdown("#### 3. What kind of cinephile are you?")
    cinephile = st.slider("Mainstream ↔ Hidden gems", 0.0, 1.0, 0.5, 0.05,
                          help="0 = safe popular picks, 1 = surprise me with films nobody talks about")

    ready = len(liked) >= MIN_LIKES
    if not ready:
        st.info(f"Pick {MIN_LIKES - len(liked)} more film(s) you loved to continue.")
    if st.button("Build my profile", type="primary", disabled=not ready):
        done = api("POST", "/onboarding", json={"user_id": user["user_id"], "liked_ids": liked, "disliked_ids": disliked,
                                                 "genres_liked": genres_liked, "genres_avoided": genres_avoided,
                                                 "cinephile": cinephile})
        if done:
            st.session_state.pop("picked", None)
            st.rerun()


def chat_tab(user: dict):
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_id", None)
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🎬"):
            st.write(f"Hey {user.get('pseudo') or 'there'}! What do you feel like watching? "
                     "You can also tell me about films you loved or hated, I'll remember.")
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="🎬" if message["role"] == "assistant" else None):
            st.markdown(message["content"])
            if message.get("films"):
                film_grid(message["films"], user["user_id"], f"chat{index}", conversation_id=message.get("conversation_id"))
    prompt = st.chat_input("e.g. an action film for tonight, but not too violent")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking about your taste..."):
            turn = api("POST", "/chat", json={"user_id": user["user_id"], "message": prompt,
                                               "conversation_id": st.session_state.conversation_id})
        if turn:
            st.session_state.conversation_id = turn["conversation_id"]
            st.session_state.messages.append({"role": "assistant", "content": turn["reply"],
                                              "films": turn["recommendations"][:5],
                                              "conversation_id": turn["conversation_id"]})
        st.rerun()


def for_you_tab(user: dict):
    genres = api("GET", "/genres") or []
    left, right = st.columns([3, 1])
    genre = left.multiselect("Filter by genre", genres, key="for_you_genres")
    k = right.number_input("How many", 5, 30, 10, 5)
    result = api("GET", "/recommend", params={"user_id": user["user_id"], "k": int(k), "genre": genre or None})
    if result:
        film_grid(result["recommendations"], user["user_id"], "foryou")


def profile_tab(user: dict):
    left, middle, right = st.columns(3)
    left.metric("Interactions", user["n_interactions"])
    middle.metric("Films liked", len(user["liked_ids"]))
    right.metric("Films disliked", len(user["disliked_ids"]))
    st.write("**Genres you love:** " + (", ".join(user["genres_liked"]) or "-"))
    st.write("**Genres you avoid:** " + (", ".join(user["genres_avoided"]) or "-"))
    st.markdown("#### Films you liked")
    st.write(", ".join(label(f) for f in user.get("liked", [])) or "-")
    if user.get("disliked"):
        st.markdown("#### Films you didn't like")
        st.write(", ".join(label(f) for f in user["disliked"]))
    if user["facts"]:
        st.markdown("#### What I learned from our chats")
        st.dataframe([{"what": f["key"], "value": f["value"], "when": f["ts"]} for f in user["facts"]],
                     hide_index=True)


def main():
    if "user_id" not in st.session_state:
        sign_in()
        return
    user = api("GET", f"/profile/{st.session_state.user_id}")
    if not user:
        st.session_state.pop("user_id")
        st.rerun()
    with st.sidebar:
        st.markdown(f"### 🎬 {user.get('pseudo') or 'User'}")
        st.caption(f"User ID: {user['user_id']}")
        level = st.slider("Mainstream ↔ Hidden gems", 0.0, 1.0, float(user["cinephile_level"]), 0.05)
        if abs(level - user["cinephile_level"]) > 1e-6:
            api("PATCH", f"/profile/{user['user_id']}", json={"cinephile": level})
            st.rerun()
        if st.button("New conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
        if st.button("Sign out"):
            st.session_state.clear()
            st.rerun()
    if not user["onboarding_done"]:
        questionnaire(user)
        return
    chat, for_you, me = st.tabs(["💬 Chat", "🎯 For you", "👤 My profile"])
    with chat:
        chat_tab(user)
    with for_you:
        for_you_tab(user)
    with me:
        profile_tab(user)


main()
