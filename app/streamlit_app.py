import hashlib
import hmac
import os
import time
from html import escape

import requests
import streamlit as st

# 127.0.0.1 and not localhost: on Windows, localhost tries IPv6 first and every call then waits 2 s
API_URL = os.environ.get("CINERADAR_API", "http://127.0.0.1:8000")
# Online only (empty in local use): the code people type to open the app, and the secret the app sends to the API
ACCESS_CODE = os.environ.get("CINERADAR_ACCESS_CODE", "")
API_TOKEN = os.environ.get("CINERADAR_API_TOKEN", "")
MAX_WRONG_CODES = 8  # per visitor and per 10 minutes
MIN_LIKES = 5
BUTTONS = [("Liked", "liked", "Liked it"), ("Disliked", "disliked", "Didn't like it"),
           ("Seen", "seen", "Already seen"), ("Not for me", "not_interested", "Not interested")]
EXAMPLES = ["Un film comme Heat, mais moins connu", "Something like Blade Runner",
            "Un thriller psychologique, plutôt une pépite", "A feel-good film, nothing violent"]

st.set_page_config(page_title="CinéRadar", page_icon=":material/movie:", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(1100px 520px at 75% -8%, rgba(124, 58, 237, .20), transparent 62%),
                    radial-gradient(900px 500px at -10% 110%, rgba(236, 72, 153, .07), transparent 60%), #0e0e15}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #171624, #101018); border-right: 1px solid rgba(139, 92, 246, .15)}
.block-container {padding-top: 3.6rem; max-width: 1400px}
[class*="st-key-card-"] {background: linear-gradient(180deg, #1c1a2b, #13131c); border-color: rgba(139, 92, 246, .16) !important;
                         box-shadow: 0 6px 18px rgba(0, 0, 0, .35); transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease}
[class*="st-key-card-"]:hover {transform: translateY(-4px); border-color: rgba(139, 92, 246, .55) !important;
                               box-shadow: 0 16px 34px rgba(0, 0, 0, .55), 0 0 22px rgba(139, 92, 246, .18)}
[class*="st-key-card-"]:hover .cr-poster {transform: scale(1.04)}
.cr-brand {font-weight: 800; background: linear-gradient(90deg, #ddd6fe, #a78bfa 45%, #f0abfc); -webkit-background-clip: text;
           background-clip: text; color: transparent}
.cr-eyebrow {font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #a78bfa; margin-bottom: -.4rem}
.cr-scale {display: flex; justify-content: space-between; font-size: .72rem; opacity: .65; margin-top: -.6rem}
.cr-level {font-size: .8rem; color: #c4b5fd; margin: .2rem 0 1rem}
.cr-bar {display: grid; grid-template-columns: 110px 1fr 34px; align-items: center; gap: 10px; font-size: .85rem; margin: 6px 0; max-width: 620px}
.cr-bar > div {height: 8px; border-radius: 999px; background: rgba(255, 255, 255, .06); overflow: hidden}
.cr-bar > div > span {display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #7c3aed, #e879f9)}
.cr-strip {display: grid; grid-template-columns: repeat(8, 1fr); gap: 12px; margin: 2rem 0 1rem; opacity: .9;
           -webkit-mask-image: linear-gradient(180deg, #000 55%, transparent); mask-image: linear-gradient(180deg, #000 55%, transparent)}
.cr-strip img {width: 100%; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 8px; box-shadow: 0 10px 24px rgba(0, 0, 0, .5)}
.cr-frame {position: relative; overflow: hidden; border-radius: 8px; box-shadow: 0 8px 20px rgba(0, 0, 0, .45)}
.cr-poster {width: 100%; aspect-ratio: 2 / 3; object-fit: cover; border-radius: 8px; display: block; transition: transform .35s ease}
.cr-noposter {background: linear-gradient(160deg, #2b2440, #15151f); display: flex; align-items: center;
              justify-content: center; text-align: center; padding: 10px; font-weight: 600}
.cr-gem {position: absolute; top: 8px; left: 8px; background: #7c3aed; color: #fff; font-size: .68rem;
         font-weight: 700; letter-spacing: .05em; text-transform: uppercase; padding: 3px 8px; border-radius: 999px;
         box-shadow: 0 2px 10px rgba(124, 58, 237, .6)}
.cr-title {font-weight: 600; line-height: 1.25; margin: .55rem 0 .2rem; min-height: 2.5em; display: -webkit-box;
           -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden}
.cr-meta {font-size: .8rem; opacity: .7; white-space: nowrap; overflow: hidden; text-overflow: ellipsis}
.cr-because {font-size: .8rem; color: #c4b5fd; margin-top: .25rem; display: -webkit-box; -webkit-line-clamp: 2;
             -webkit-box-orient: vertical; overflow: hidden}
.cr-shelf {display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 14px; margin: .5rem 0 1.5rem}
.cr-shelf .cr-title {font-size: .8rem; min-height: 0}
[data-testid="stButtonGroup"] button {padding: 1px 9px; min-height: 0; font-size: .75rem}
.cr-hero h1 {font-size: 3.6rem; margin-bottom: 0; line-height: 1.1}
.cr-hero p {font-size: 1.15rem; opacity: .8}
/* compact card (phones): small poster on the left, text on the right */
.cr-row {display: grid; grid-template-columns: 84px 1fr; gap: 12px; align-items: start}
.cr-row .cr-title {margin-top: 0; min-height: 0; font-size: .95rem}
.cr-row .cr-meta {white-space: normal}
.cr-row .cr-gem {top: 4px; left: 4px; font-size: .55rem; padding: 2px 5px}
@media (max-width: 900px) {
  .block-container {padding: 3.2rem 1rem 5rem}
  .cr-hero h1 {font-size: 2.4rem}
  .cr-hero p {font-size: 1rem}
  .cr-strip {grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 1rem 0 .5rem}
  .cr-strip img:nth-child(n+5) {display: none}
  /* film grids: 2 or 3 per row instead of 5 squeezed ones */
  [class*="st-key-grid-"] [data-testid="stHorizontalBlock"] {flex-wrap: wrap; gap: .8rem}
  [class*="st-key-grid-"] [data-testid="stColumn"] {flex: 1 1 calc(33% - .8rem) !important; min-width: calc(33% - .8rem) !important}
  .cr-shelf {grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); gap: 10px}
  [data-testid="stMetricValue"] {font-size: 1.5rem}
}
@media (max-width: 640px) {
  [class*="st-key-grid-"] [data-testid="stColumn"] {flex: 1 1 calc(50% - .8rem) !important; min-width: calc(50% - .8rem) !important}
  .cr-hero h1 {font-size: 2rem}
  h1 {font-size: 1.9rem !important}
  h4 {font-size: 1.1rem !important}
  .cr-bar {grid-template-columns: 90px 1fr 28px}
  [data-testid="stChatMessage"] {padding: .5rem .25rem}
}
/* phone tabs and profile numbers: always side by side */
.st-key-mobilenav [data-testid="stHorizontalBlock"], .st-key-metrics [data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: .4rem}
.st-key-mobilenav [data-testid="stColumn"], .st-key-metrics [data-testid="stColumn"] {min-width: 0 !important; flex: 1 1 0 !important}
.st-key-mobilenav a {justify-content: center; border: 1px solid rgba(139, 92, 246, .25); border-radius: 10px;
                     padding: .35rem .2rem; background: rgba(139, 92, 246, .08)}
.st-key-mobilenav a[aria-current="page"], .st-key-mobilenav a[data-active="true"] {background: rgba(139, 92, 246, .35)}
@media (hover: none) {
  [class*="st-key-card-"]:hover {transform: none}
  [class*="st-key-card-"]:hover .cr-poster {transform: none}
  [data-testid="stButtonGroup"] button {padding: 6px 12px; font-size: .85rem}  /* bigger touch targets */
}
</style>
""", unsafe_allow_html=True)


def is_phone() -> bool:
    # Phones get the compact view by default, the sidebar toggle can switch it either way
    agent = (st.context.headers or {}).get("User-Agent", "") if hasattr(st, "context") else ""
    return any(word in agent for word in ("iPhone", "Android", "Mobile", "iPod"))


def compact() -> bool:
    if "compact" not in st.session_state:
        st.session_state.compact = is_phone()
    return st.session_state.compact


@st.cache_resource
def http() -> requests.Session:
    session = requests.Session()  # keeps the connection to the API open between calls
    if API_TOKEN:
        session.headers["X-CineRadar-Token"] = API_TOKEN
    return session


def visitor() -> str:
    # Who is calling, so the API limits each visitor and not everybody at once (hashed, never stored)
    headers = st.context.headers or {}
    ip = (headers.get("X-Forwarded-For") or "").split(",")[0].strip() or getattr(st.context, "ip_address", "") or ""
    return hashlib.sha256(ip.encode()).hexdigest()[:16] if ip else "local"


def api(method: str, path: str, **kwargs):
    try:
        response = http().request(method, f"{API_URL}{path}", timeout=30,
                                  headers={"X-CineRadar-Client": visitor()}, **kwargs)
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


def film_row(film: dict, user_id: int, key: str, conversation_id=None):
    # Compact card for phones: one film per line, small poster, text beside it
    with st.container(border=True, key=f"card-{key}-{film['movie_id']}"):
        text = f'<div class="cr-title">{escape(label(film))}</div><div class="cr-meta">{escape(meta(film))}</div>'
        if film.get("directors"):
            text += f'<div class="cr-meta">by {escape(film["directors"].split(",")[0])}</div>'
        if film.get("because_title"):
            text += f'<div class="cr-because">Because you liked {escape(film["because_title"])}</div>'
        st.markdown(f'<div class="cr-row">{poster(film)}<div>{text}</div></div>', unsafe_allow_html=True)
        if film.get("overview"):
            with st.expander("Summary"):
                st.write(film["overview"])
        opinion(film, user_id, key, conversation_id)


def film_card(film: dict, user_id: int, key: str, conversation_id=None):
    with st.container(border=True, key=f"card-{key}-{film['movie_id']}"):
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
    if compact():
        for film in films:
            film_row(film, user_id, key, conversation_id)
        return
    with st.container(key=f"grid-{key}"):  # the CSS wraps this grid on narrow screens
        for start in range(0, len(films), per_row):
            columns = st.columns(per_row)
            for column, film in zip(columns, films[start:start + per_row]):
                with column:
                    film_card(film, user_id, key, conversation_id)


def shelf(films: list[dict]):
    # Small posters without buttons, for the profile page
    items = "".join(f'<div>{poster(f)}<div class="cr-title">{escape(label(f))}</div></div>' for f in films)
    st.markdown(f'<div class="cr-shelf">{items}</div>', unsafe_allow_html=True)


def header(eyebrow: str, title: str, caption: str):
    st.markdown(f'<div class="cr-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(caption)


def level_text(level: float) -> str:
    if level < 0.3:
        return "Mostly well-known crowd pleasers"
    if level < 0.6:
        return "A mix of classics and lesser-known films"
    if level < 0.85:
        return "Mostly lesser-known films that people loved"
    return "Films almost nobody has seen, but that people loved"


# Pages before sign-in

def sign_in():
    st.markdown('<div class="cr-hero"><h1><span class="cr-brand">CinéRadar</span></h1>'
                "<p>Your film-loving friend who finds the hidden gems you would never find alone</p></div>",
                unsafe_allow_html=True)
    films = [f for f in (static("/movies/popular", limit=40) or []) if f.get("poster_url")][:4 if compact() else 8]
    if films:
        st.markdown('<div class="cr-strip">' + "".join(f'<img src="{escape(f["poster_url"])}" alt="">' for f in films)
                    + "</div>", unsafe_allow_html=True)
        st.caption("Everybody has seen these. CinéRadar looks for the films you would love next and nobody told you about")
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

def quick_settings():
    # On a phone the sidebar is hidden behind a menu: the slider and the main actions sit at the top
    if not compact():
        return
    user = st.session_state.user
    with st.popover(f"Settings · hidden gems {user['cinephile_level']:.0%}", icon=":material/tune:",
                    use_container_width=True):
        level_slider(user, "level_top")
        st.button("New conversation", on_click=new_conversation, use_container_width=True, key="new_top")


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
    quick_settings()
    prompt = st.chat_input("e.g. an action film for tonight, but not too violent") or st.session_state.pop("pending", None)
    messages = st.session_state.messages

    with st.chat_message("assistant"):
        st.write(f"Hey {user.get('pseudo') or 'there'}! What do you feel like watching? "
                 "You can also tell me about films you loved or hated, I'll remember")
        if not messages and not prompt:
            st.caption("Try one of these")
            columns = st.columns(1 if compact() else 2)
            for index, text in enumerate(EXAMPLES):
                columns[index % len(columns)].button(text, on_click=send_example, args=(text,), use_container_width=True)

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
    header("Your picks", "Picked for you",
           "From your profile and everything you told me. Move the slider in the settings to get more hidden gems")
    quick_settings()
    if compact():
        genre = st.multiselect("Filter by genre", static("/genres") or [], key="for_you_genres")
        k = 8
    else:
        left, right = st.columns([4, 1])
        genre = left.multiselect("Filter by genre", static("/genres") or [], key="for_you_genres")
        k = right.number_input("How many", 5, 30, 10, 5)
    with st.spinner("Picking films..."):
        result = api("GET", "/recommend", params={"user_id": user["user_id"], "k": int(k), "genre": genre or None})
    if result:
        film_grid(result["recommendations"], user["user_id"], "foryou")


def profile_page():
    user = st.session_state.user
    header("Your taste", "My profile", "Everything I know about your taste. It grows with each message and each click")
    metrics = [("Interactions", user["n_interactions"]), ("Films liked", len(user["liked_ids"])),
               ("Films disliked", len(user["disliked_ids"])), ("Hidden gems", f"{user['cinephile_level']:.0%}")]
    rows = [metrics[:2], metrics[2:]] if compact() else [metrics]  # 2 x 2 on a phone
    with st.container(key="metrics"):
        for row in rows:
            for column, (name, value) in zip(st.columns(len(row)), row):
                column.metric(name, value)
    left, right = st.columns(2)
    with left:
        st.markdown("**Genres you love**")
        st.markdown(" ".join(f":violet-badge[{g}]" for g in user["genres_liked"]) or "None yet")
    with right:
        st.markdown("**Genres you avoid**")
        st.markdown(" ".join(f":gray-badge[{g}]" for g in user["genres_avoided"]) or "None yet")
    liked = user.get("liked", [])
    counts = {}
    for film in liked:
        for genre in (film.get("genres") or "").split(","):
            if genre.strip():
                counts[genre.strip()] = counts.get(genre.strip(), 0) + 1
    if counts:
        st.markdown("#### The genres of the films you liked")
        top = sorted(counts.items(), key=lambda item: -item[1])[:6]
        st.markdown("".join(f'<div class="cr-bar">{escape(g)}<div><span style="width:{100 * n / len(liked):.0f}%"></span></div>'
                            f"{n}</div>" for g, n in top), unsafe_allow_html=True)
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


def set_level(key: str = "level"):
    level = st.session_state[key]
    api("PATCH", f"/profile/{st.session_state.user_id}", json={"cinephile": level})
    for other in ("level", "level_top"):  # the sidebar and the phone settings show the same value
        st.session_state[other] = level
    st.session_state.user["cinephile_level"] = level


def level_slider(user: dict, key: str):
    st.slider("What kind of films?", 0.0, 1.0, float(user["cinephile_level"]), 0.05,
              key=key, on_change=set_level, args=(key,),
              help="Move right and I will suggest films almost nobody has seen, but that people loved")
    st.markdown('<div class="cr-scale"><span>Mainstream</span><span>Hidden gems</span></div>'
                f'<div class="cr-level">{level_text(user["cinephile_level"])}</div>', unsafe_allow_html=True)


def new_conversation():
    st.session_state.messages = []
    st.session_state.conversation_id = None


def sidebar(user: dict):
    with st.sidebar:
        st.markdown('<h2 style="margin-bottom:0"><span class="cr-brand">CinéRadar</span></h2>', unsafe_allow_html=True)
        st.caption(f"Signed in as **{user.get('pseudo') or 'User'}** (user ID {user['user_id']})")
        if user["onboarding_done"]:
            level_slider(user, "level")
            st.button("New conversation", on_click=new_conversation, use_container_width=True)
        st.toggle("Compact view (phone)", key="compact", help="One film per line with small posters, "
                  "turned on automatically on phones")
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()


@st.cache_resource
def wrong_codes() -> dict:
    return {}  # visitor -> times of the last wrong codes, shared by every session of the server


def code_is_right(given: str) -> bool:
    return hmac.compare_digest(given.strip().encode(), ACCESS_CODE.encode())


def locked_out() -> bool:
    recent = [t for t in wrong_codes().get(visitor(), []) if time.time() - t < 600]
    wrong_codes()[visitor()] = recent
    return len(recent) >= MAX_WRONG_CODES


def access_gate() -> bool:
    """Online, the app stays closed until the right code is given, typed or in the link (?code=...)."""
    if not ACCESS_CODE and os.environ.get("CINERADAR_REQUIRE_CODE"):
        # online without a code configured: stay closed rather than open to everybody
        st.error("CinéRadar is not configured yet: set the CINERADAR_ACCESS_CODE secret")
        return False
    if not ACCESS_CODE or st.session_state.get("access_ok"):
        return True
    given = st.query_params.get("code", "")
    if given and not locked_out():
        if code_is_right(given):
            st.session_state.access_ok = True
            del st.query_params["code"]  # the code does not stay in the address bar
            return True
        wrong_codes().setdefault(visitor(), []).append(time.time())
    st.markdown('<div class="cr-hero"><h1><span class="cr-brand">CinéRadar</span></h1>'
                "<p>Private beta: enter the access code</p></div>", unsafe_allow_html=True)
    if locked_out():
        st.error("Too many wrong codes, try again in 10 minutes")
        return False
    with st.form("code"):
        typed = st.text_input("Access code", type="password")
        if st.form_submit_button("Open", type="primary"):
            if code_is_right(typed):
                st.session_state.access_ok = True
                st.rerun()
            wrong_codes().setdefault(visitor(), []).append(time.time())
            st.error("Wrong code")
    return False


def sign_in_from_link():
    # ?user=12 in the link (the github.io page adds it) signs in directly, the profile must exist
    wanted = st.query_params.get("user", "")
    if "user_id" not in st.session_state and wanted.isdigit():
        if api("GET", f"/profile/{int(wanted)}"):
            st.session_state.user_id = int(wanted)


def main():
    compact()  # decide the view once per session, before any widget reads it
    if not access_gate():
        return
    sign_in_from_link()
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
    if not compact():
        st.navigation(pages, position="top").run()
        return
    # On a phone the top menu folds into the sidebar: a row of three tabs stays visible instead
    page = st.navigation(pages, position="hidden")
    with st.container(key="mobilenav"):
        for column, target in zip(st.columns(3), pages):
            column.page_link(target, label=target.title.replace("My profile", "Profile"), icon=target.icon,
                             use_container_width=True)
    page.run()


main()
