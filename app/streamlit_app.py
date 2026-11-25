import os
from html import escape

import requests
import streamlit as st

# 127.0.0.1 and not localhost: on Windows, localhost tries IPv6 first and every call then waits 2 s
API_URL = os.environ.get("CINERADAR_API", "http://127.0.0.1:8000")
MIN_LIKES = 5
BUTTONS = [("Liked", "liked", "Liked it"), ("Disliked", "disliked", "Didn't like it"),
           ("Seen", "seen", "Already seen"), ("Not for me", "not_interested", "Not interested")]
EXAMPLES = ["Un film comme Heat, mais moins connu", "Something like Blade Runner",
            "Un thriller psychologique, plutôt une pépite", "A feel-good film, nothing violent"]

st.set_page_config(page_title="CinéRadar", page_icon=":material/movie:", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px}
.cr-frame {position: relative}
.cr-poster {width: 100%; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 8px; display: block}
.cr-noposter {background: linear-gradient(160deg, #2b2440, #15151f); display: flex; align-items: center;
              justify-content: center; text-align: center; padding: 10px; font-weight: 600}
.cr-gem {position: absolute; top: 8px; left: 8px; background: #7c3aed; color: #fff; font-size: .68rem;
         font-weight: 700; letter-spacing: .05em; text-transform: uppercase; padding: 3px 8px; border-radius: 999px;
         box-shadow: 0 2px 6px rgba(0,0,0,.4)}
.cr-title {font-weight: 600; line-height: 1.25; margin: .55rem 0 .2rem; min-height: 2.5em; display: -webkit-box;
           -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden}
.cr-meta {font-size: .8rem; opacity: .7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis}
.cr-because {font-size: .8rem; color: #c4b5fd; margin-top: .25rem; display: -webkit-box; -webkit-line-clamp: 2;
             -webkit-box-orient: vertical; overflow: hidden}
.cr-shelf {display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 14px; margin: .5rem 0 1.5rem}
.cr-shelf .cr-title {font-size: .8rem; min-height: 0}
[data-testid="stButtonGroup"] button {padding: 1px 9px; min-height: 0; font-size: .75rem}
.cr-hero h1 {font-size: 3rem; margin-bottom: 0}
.cr-hero p {font-size: 1.15rem; opacity: .8}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def http() -> requests.Session:
    return requests.Session()  # keeps the connection to the API open between calls


def api(method: str, path: str, **kwargs):
    try:
        response = http().request(method, f"{API_URL}{path}", timeout=30, **kwargs)
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


@st.cache_data(ttl=600, show_spinner=False)
def _cached_get(path: str, params: dict | None):
    response = http().get(f"{API_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def static(path: str, **params):
    # GET for data that does not depend on the user (genres, popular films), cached for 10 minutes
    try:
        return _cached_get(path, params or None)
    except requests.RequestException:
        return api("GET", path, params=params)


def label(film: dict) -> str:
    year = f" ({film['year']})" if film.get("year") else ""
    return f"{film['title']}{year}"


def poster(film: dict) -> str:
    gem = '<span class="cr-gem">hidden gem</span>' if film.get("hidden_gem") else ""
    if film.get("poster_url"):
        image = f'<img class="cr-poster" src="{escape(film["poster_url"])}" loading="lazy" alt="">'
    else:
        image = f'<div class="cr-poster cr-noposter">{escape(film["title"])}</div>'
    return f'<div class="cr-frame">{image}{gem}</div>'


def meta(film: dict) -> str:
    parts = [str(film["year"])] if film.get("year") else []
    parts += [g.strip() for g in (film.get("genres") or "").split(",")[:2] if g.strip()]
    if film.get("mean_rating"):
        parts.append(f"{film['mean_rating']}/5")
    return " · ".join(parts)


@st.fragment
def opinion(film: dict, user_id: int, key: str, conversation_id=None):
    # A fragment: clicking a button reruns only this card, not the whole page
    widget = f"{key}-opinion-{film['movie_id']}"
    choice = st.pills("Your opinion", [b[0] for b in BUTTONS], key=widget, label_visibility="collapsed")
    if choice and st.session_state.get(f"{widget}-saved") != choice:
        _, kind, tip = next(b for b in BUTTONS if b[0] == choice)
        api("POST", "/log_interaction", json={"user_id": user_id, "movie_id": film["movie_id"], "type": kind,
                                              "source": "chat" if conversation_id else "click",
                                              "conversation_id": conversation_id})
        st.session_state[f"{widget}-saved"] = choice
        st.toast(f"{tip}: {film['title']} saved to your profile")


def film_card(film: dict, user_id: int, key: str, conversation_id=None):
    with st.container(border=True):
        html = poster(film) + f'<div class="cr-title">{escape(label(film))}</div><div class="cr-meta">{escape(meta(film))}</div>'
        if film.get("directors"):
            html += f'<div class="cr-meta">by {escape(film["directors"].split(",")[0])}</div>'
        if film.get("because_title"):
            html += f'<div class="cr-because">Because you liked {escape(film["because_title"])}</div>'
        st.markdown(html, unsafe_allow_html=True)
        if film.get("overview"):
            with st.expander("Summary"):
                st.write(film["overview"])
        opinion(film, user_id, key, conversation_id)


def film_grid(films: list[dict], user_id: int, key: str, per_row: int = 5, conversation_id=None):
    for start in range(0, len(films), per_row):
        columns = st.columns(per_row)
        for column, film in zip(columns, films[start:start + per_row]):
            with column:
                film_card(film, user_id, key, conversation_id)


def shelf(films: list[dict]):
    # Small posters without buttons, for the profile page
    items = "".join(f'<div>{poster(f)}<div class="cr-title">{escape(label(f))}</div></div>' for f in films)
    st.markdown(f'<div class="cr-shelf">{items}</div>', unsafe_allow_html=True)


# Pages before sign-in

def sign_in():
    st.markdown('<div class="cr-hero"><h1>CinéRadar</h1>'
                "<p>Your film-loving friend who finds the hidden gems you would never find alone</p></div>",
                unsafe_allow_html=True)
    st.write("")
    new, existing = st.columns(2, gap="large")
    with new.container(border=True):
        st.markdown("#### New here?")
        pseudo = st.text_input("Pick a nickname")
        if st.button("Start", type="primary", disabled=not pseudo.strip()):
            user = api("POST", "/users", json={"pseudo": pseudo.strip()})
            if user:
                st.session_state.user_id = user["user_id"]
                st.rerun()
    with existing.container(border=True):
        st.markdown("#### Already have a profile?")
        user_id = st.number_input("Your user ID", min_value=1, step=1)
        if st.button("Continue"):
            if api("GET", f"/profile/{int(user_id)}"):
                st.session_state.user_id = int(user_id)
                st.rerun()
    st.caption("Tell it a few films you loved, then ask for anything, in French or in English")


def questionnaire():
    user = st.session_state.user
    st.title(f"Welcome {user.get('pseudo') or ''}")
    st.write("A few questions so I can get to know your taste, and every answer you give later keeps teaching me")
    if "picked" not in st.session_state:
        st.session_state.picked = {}
    genres = static("/genres") or []

    st.markdown("#### 1. Genres")
    left, right = st.columns(2)
    genres_liked = left.multiselect("Genres you love", genres)
    genres_avoided = right.multiselect("Genres you avoid", [g for g in genres if g not in genres_liked])

    st.markdown(f"#### 2. Films you loved (at least {MIN_LIKES}, ideally 10)")
    suggestions = static("/movies/popular", limit=120, genre=genres_liked or None) or []
    query = st.text_input("Search a film", placeholder="e.g. Heat, Amelie, Spirited Away")
    if query.strip():
        suggestions = (static("/movies/search", q=query.strip(), limit=20) or []) + suggestions
    options = {f["movie_id"]: f for f in list(st.session_state.picked.values()) + suggestions}
    liked = st.multiselect("Pick the films you loved", list(options), format_func=lambda m: label(options[m]),
                           key="liked_select")
    for movie_id in liked:
        st.session_state.picked[movie_id] = options[movie_id]
    if liked:
        shelf([options[m] for m in liked])
    disliked = st.multiselect("And a few films you really didn't like (optional)",
                              [m for m in options if m not in liked], format_func=lambda m: label(options[m]))

    st.markdown("#### 3. What kind of cinephile are you?")
    cinephile = st.slider("Mainstream ↔ Hidden gems", 0.0, 1.0, 0.5, 0.05,
                          help="0 gives safe popular picks, 1 gives films nobody talks about")

    ready = len(liked) >= MIN_LIKES
    if not ready:
        st.info(f"Pick {MIN_LIKES - len(liked)} more film(s) you loved to continue")
    if st.button("Build my profile", type="primary", disabled=not ready):
        done = api("POST", "/onboarding", json={"user_id": user["user_id"], "liked_ids": liked, "disliked_ids": disliked,
                                                 "genres_liked": genres_liked, "genres_avoided": genres_avoided,
                                                 "cinephile": cinephile})
        if done:
            st.session_state.pop("picked", None)
            st.rerun()


# Pages after sign-in

def send_example(text: str):
    st.session_state.pending = text


def show_answer(message: dict, index: int, latest: bool):
    st.markdown(message["content"].replace("\n", "  \n"))  # keep the line breaks of the reply
    films = message.get("films") or []
    if not films:
        return
    user_id = st.session_state.user["user_id"]
    if latest:
        film_grid(films, user_id, f"chat{index}", conversation_id=message.get("conversation_id"))
    else:
        with st.expander(f"The {len(films)} films of this answer"):
            film_grid(films, user_id, f"chat{index}", conversation_id=message.get("conversation_id"))


def chat_page():
    user = st.session_state.user
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_id", None)
    prompt = st.chat_input("e.g. an action film for tonight, but not too violent") or st.session_state.pop("pending", None)
    messages = st.session_state.messages

    with st.chat_message("assistant"):
        st.write(f"Hey {user.get('pseudo') or 'there'}! What do you feel like watching? "
                 "You can also tell me about films you loved or hated, I'll remember")
        if not messages and not prompt:
            st.caption("Try one of these")
            columns = st.columns(2)
            for index, text in enumerate(EXAMPLES):
                columns[index % 2].button(text, on_click=send_example, args=(text,), use_container_width=True)

    last_answer = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
    for index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                show_answer(message, index, latest=index == last_answer and not prompt)

    if prompt:
        # Shown at once and without a page reload, the answer then appears under it
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking about your taste..."):
                turn = api("POST", "/chat", json={"user_id": user["user_id"], "message": prompt,
                                                   "conversation_id": st.session_state.conversation_id})
            if turn:
                st.session_state.conversation_id = turn["conversation_id"]
                answer = {"role": "assistant", "content": turn["reply"], "films": turn["recommendations"],
                          "conversation_id": turn["conversation_id"]}
                messages.append(answer)
                show_answer(answer, len(messages) - 1, latest=True)


def for_you_page():
    user = st.session_state.user
    st.title("Picked for you")
    st.caption("From your profile and everything you told me. Move the slider in the sidebar to get more hidden gems")
    left, right = st.columns([4, 1])
    genre = left.multiselect("Filter by genre", static("/genres") or [], key="for_you_genres")
    k = right.number_input("How many", 5, 30, 10, 5)
    with st.spinner("Picking films..."):
        result = api("GET", "/recommend", params={"user_id": user["user_id"], "k": int(k), "genre": genre or None})
    if result:
        film_grid(result["recommendations"], user["user_id"], "foryou")


def profile_page():
    user = st.session_state.user
    st.title("My profile")
    st.caption("Everything I know about your taste. It grows with each message and each click")
    columns = st.columns(4)
    columns[0].metric("Interactions", user["n_interactions"])
    columns[1].metric("Films liked", len(user["liked_ids"]))
    columns[2].metric("Films disliked", len(user["disliked_ids"]))
    columns[3].metric("Hidden gems level", f"{user['cinephile_level']:.0%}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Genres you love**")
        st.markdown(" ".join(f":violet-badge[{g}]" for g in user["genres_liked"]) or "None yet")
    with right:
        st.markdown("**Genres you avoid**")
        st.markdown(" ".join(f":gray-badge[{g}]" for g in user["genres_avoided"]) or "None yet")
    st.markdown("#### Films you liked")
    if user.get("liked"):
        shelf(user["liked"])
    else:
        st.write("None yet")
    if user.get("disliked"):
        st.markdown("#### Films you didn't like")
        shelf(user["disliked"])
    if user.get("seen"):
        st.markdown("#### Films you have already seen")
        shelf(user["seen"])
    if user["facts"]:
        st.markdown("#### What I learned from our chats")
        st.dataframe([{"what": f["key"], "value": f["value"], "when": f["ts"]} for f in user["facts"]],
                     hide_index=True, use_container_width=True)


def set_level():
    api("PATCH", f"/profile/{st.session_state.user_id}", json={"cinephile": st.session_state.level})


def new_conversation():
    st.session_state.messages = []
    st.session_state.conversation_id = None


def sidebar(user: dict):
    with st.sidebar:
        st.markdown("## CinéRadar")
        st.caption(f"Signed in as **{user.get('pseudo') or 'User'}** (user ID {user['user_id']})")
        if user["onboarding_done"]:
            st.slider("Mainstream ↔ Hidden gems", 0.0, 1.0, float(user["cinephile_level"]), 0.05,
                      key="level", on_change=set_level,
                      help="Move right and I will suggest films almost nobody has seen, but that people loved")
            st.button("New conversation", on_click=new_conversation, use_container_width=True)
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()


def main():
    if "user_id" not in st.session_state:
        st.navigation([st.Page(sign_in, title="Welcome")], position="hidden").run()
        return
    user = api("GET", f"/profile/{st.session_state.user_id}")
    if not user:
        st.session_state.pop("user_id")
        st.rerun()
    st.session_state.user = user
    sidebar(user)
    if not user["onboarding_done"]:
        st.navigation([st.Page(questionnaire, title="Your taste")], position="hidden").run()
        return
    pages = [st.Page(chat_page, title="Chat", icon=":material/chat:", default=True),
             st.Page(for_you_page, title="For you", icon=":material/movie:", url_path="for-you"),
             st.Page(profile_page, title="My profile", icon=":material/person:", url_path="profile")]
    st.navigation(pages, position="top").run()


main()
