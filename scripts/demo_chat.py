"""Démo terminal de la partie 3.

    python scripts/demo_chat.py            # chat interactif (profil SF grand public)
    python scripts/demo_chat.py --profils  # critère « terminé » : 2 profils, même demande

Sans clé API, tout tourne en mode règles + phrases toutes faites.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import agent, client  # noqa: E402
from src.reco import engine  # noqa: E402
from src.reco.catalog import get_catalog  # noqa: E402

cat = get_catalog()
T = cat.find_title
mem = engine.DEMO_MEMORY
mem.onboarding(1, [T(x) for x in ["The Matrix", "Inception", "Interstellar", "Avengers: Endgame"]],
               genres_liked=["Sci-Fi"], cinephile=0.2)
mem.onboarding(2, [T(x) for x in ["Le Samouraï", "Oldboy", "Heat", "Memories of Murder"]],
               genres_disliked=["Romance"], cinephile=0.9)


def show(recos):
    for r in recos:
        m = cat.get(r["movie_id"])
        b = cat.get(r["because_movie_id"])["title"] if r["because_movie_id"] else "-"
        print(f"   {m['title']:<38} {m['year']}  pop={m['popularity']:.1f}  score={r['score']:.2f}  car: {b}")


if "--profils" in sys.argv:
    msg = "Je veux un film d'action"
    for uid, label in [(1, "A — fan de SF grand public (curseur 0.2)"), (2, "B — cinéphile polar/Asie (curseur 0.9)")]:
        out = agent.chat_turn(uid, msg, [])
        print(f"\n=== Profil {label} : « {msg} »")
        show(out["recos"])
        print("\n" + out["reply"])
    sys.exit(0)

print(f"Mode LLM : {client.provider() or 'hors-ligne (règles)'} — tape 'q' pour quitter.\n")
history = []
while True:
    try:
        msg = input("toi > ").strip()
    except EOFError:
        break
    if msg.lower() in {"q", "quit", "exit"}:
        break
    out = agent.chat_turn(1, msg, history)
    print(f"\n[intention: {out['intent']['intent']} | genres={out['intent'].get('genres')} "
          f"| refs={out['intent'].get('reference_titles')}]")
    print("ami > " + out["reply"] + "\n")
    show(out["recos"])
    print()
    history += [{"role": "user", "content": msg},
                {"role": "assistant", "content": out["reply"], "intent": out["intent"],
                 "recos": [r["movie_id"] for r in out["recos"]]}]
