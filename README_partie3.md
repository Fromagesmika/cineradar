# Partie 3 — Cerveau conversationnel (LLM + classement)

Transforme un message en recommandations personnalisées et en réponse « d'ami », en s'appuyant sur la
**matrice de mémoire** de l'utilisateur (questionnaire d'accueil + clics + notes + ce qu'il dit dans le chat).

## Lancer en 1 minute

```bash
pip install numpy pandas scikit-learn pytest      # pyarrow en plus quand movies.parquet arrive
python -m pytest -q                               # 15 tests, dont le critère « terminé »
python scripts/demo_chat.py --profils             # 2 profils, même demande « un film d'action »
python scripts/demo_chat.py                       # chat interactif
python -m src.reco.ablations                      # tableau d'ablations
```

Sans clé API, tout marche en **mode règles** (mots-clés + phrases toutes faites) : c'est le mode prof.

## Fichiers

| Fichier | Rôle |
|---|---|
| `src/reco/engine.py` | `recommend(user_id, intent, k, cinephile)` : candidats, classement, pépites, « parce que » |
| `src/llm/agent.py` | `extract(message, history)`, `reply(message, profile, recos)`, `chat_turn(...)` |
| `src/reco/memory.py` | Matrice de mémoire : événements bruts → affinités, refus, lassitude. Lecture SQLite ou RAM |
| `src/reco/catalog.py` | Chargement de `movies.parquet`, recherche de titres approximative, recherche sémantique |
| `src/llm/rules.py` | Extraction sans LLM (français : genres, négations, titres, époques, « pépite », infos de profil) |
| `src/llm/client.py`, `prompts.py` | Appel LLM (Anthropic ou API compatible OpenAI : OpenAI, Groq, Mistral, Ollama) |
| `src/reco/taste_mock.py`, `mock_catalog.py` | Remplaçants des parties 1 et 2 tant qu'elles ne sont pas branchées |
| `src/reco/ablations.py` | LLM seul / + goût / + descriptions / + mémoire / + pépites |

## Branchement avec les autres parties (rien à modifier dans mon code)

- **Partie 1 (Théodore)** : dès que `data/processed/movies.parquet` existe, il remplace le mock automatiquement.
  `genres`/`tags` peuvent être des listes ou des chaînes `"Action|Crime"`. Colonne optionnelle bienvenue :
  `aliases` (titres français TMDB, ex. « Le Parrain »), sinon les titres FR ne seront pas reconnus dans le chat.
  Si `src/eval/evaluate.py` expose `evaluate(recommend_fn) -> DataFrame`, les ablations l'utilisent directement.
- **Partie 2 (Michaël)** : si `src/model/taste.py` existe avec `user_vector` et `score`, il est utilisé à la place du mock
  (forcer le mock : `TASTE_BACKEND=mock`).
- **Partie 4 (Benoît)** : si `data/app.db` existe (ou `DB_PATH=...`), la mémoire est lue dans SQLite.
  Les 4 requêtes SQL sont en haut de `SQLiteProvider` dans `memory.py` → à ajuster au schéma final lundi.

### Pour `/chat` (Benoît)

```python
from src.llm.agent import chat_turn
out = chat_turn(user_id, message, history, cinephile=slider)   # cinephile=None -> valeur en mémoire
out["reply"]            # texte de l'ami
out["recos"]            # [{"movie_id", "score", "because_movie_id"}] -> fiches avec affiche
out["intent"]           # à stocker avec le message assistant (sert pour « un autre », « plus récent »)
out["memory_updates"]   # {"interactions": [...], "profile_facts": [...]} -> à écrire dans la base
```

`history` = `[{"role": "user"|"assistant", "content": str, "intent": dict, "recos": [ids]}]`
(`intent` et `recos` seulement sur les messages assistant).

**Types d'interactions lus** : `onboarding_like`, `onboarding_dislike`, `like`, `dislike`, `chat_like`, `chat_dislike`,
`not_interested`, `seen`, `rating` (avec `value` 0.5–5), `shown`.
**Faits lus** : `genre_like`, `genre_dislike` (noms de genres MovieLens), `watch_with`, `age`, `language_pref`, `cinephile`.

## La matrice de mémoire (pour le rapport)

On a déjà deux matrices venant des parties 1 et 2 (utilisateurs × films, films × contenu). La 3e est la **mémoire
utilisateur**, qui se remplit au fil de l'utilisation :

1. **Questionnaire d'accueil** : 10 films aimés, films détestés, genres, curseur. Le profil existe avant le 1er message.
2. **Chaque événement** (clic, note, « j'ai adoré X » dans le chat) est stocké brut par la partie 4.
3. **À chaque requête**, `build_memory` recalcule : films aimés/détestés pondérés, affinité par genre
   (une ligne utilisateur × genres), films refusés, films déjà montrés. Décroissance temporelle (demi-vie de
   60 jours), sauf le questionnaire qui garde au moins 70 % de son poids.
4. **Effets sur le classement** :
   - les films vus, aimés ou refusés sont exclus ;
   - les genres détestés sont filtrés ;
   - les films proches d'un « pas intéressé » sont pénalisés ;
   - un film déjà montré plusieurs fois recule (lassitude) ;
   - les films aimés alimentent le vecteur de goût de la partie 2 et l'explication « parce que tu as aimé… ».

Un clic modifie donc la recommandation suivante (testé dans `test_un_clic_change_la_reco_suivante`).

## Score final

```
pertinence = 1.0·intention + 0.8·goût + 0.5·mémoire     (goût et mémoire ×0.5 quand l'utilisateur dit « comme X »)
final      = pertinence + 0.35·(2c−1)·(1−2·popularité)  (c = curseur : 1 pousse les pépites, 0 les classiques, 0.5 neutre)
```
Ensuite, une diversification légère (MMR sur le contenu) évite dix fois le même film.
Intention = genres demandés + mots-clés/tags + recherche sémantique (TF-IDF, ou `USE_SENTENCE_TRANSFORMERS=1`
sur Colab) + similarité aux films de référence.

## LLM

| Variable | Effet |
|---|---|
| *(rien)* ou `LLM_OFFLINE=1` | mode règles, aucun appel réseau |
| `ANTHROPIC_API_KEY` | API Anthropic (`LLM_MODEL`, défaut `claude-haiku-5-5`) |
| `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`) | API compatible OpenAI : OpenAI, Groq, Mistral, Ollama local… (`LLM_MODEL` à régler) |

Garde-fous :
- le JSON d'extraction est validé et normalisé ;
- les titres sont retrouvés dans le catalogue ;
- si le LLM rate un titre, les règles le rattrapent ;
- si la réponse « ami » cite un film absent du top fourni, elle est rejetée et remplacée par le modèle de phrases ;
- en cas d'erreur réseau ou de quota, on bascule sur les règles.

## Ablations (banc interne, catalogue mock de 79 films, 40 profils synthétiques)

| variante | Recall@10 | part pépites | couverture | Jaccard même demande (↓ = plus personnalisé) |
|---|---|---|---|---|
| LLM seul | 0.388 | 0.030 | 0.506 | 1.000 |
| + goût | 0.875 | 0.240 | 0.975 | 0.218 |
| + descriptions | 0.875 | 0.240 | 0.975 | 0.218 |
| + mémoire | 0.863 | 0.235 | 0.962 | 0.227 |
| + pépites (complet) | 0.887 | 0.297 | 0.987 | 0.221 |

⚠️ Ce banc ne sert qu'à vérifier la cohérence du système, ce ne sont **pas** les chiffres du rapport. Les profils synthétiques
sont construits par similarité de contenu, ce qui avantage les variantes à base de contenu. Sur ce petit banc, « + descriptions »
change l'ordre des listes mais pas le Recall. L'apport de la mémoire (refus, lassitude) ne se voit pas non plus, puisque ces profils
n'ont aucun historique de clics. Les vrais chiffres viendront du harnais de la partie 1 (ReDial), branché automatiquement.

## Limites connues

- Mode règles : un titre d'un seul mot doit être écrit avec une majuscule ou suivre « comme / j'ai aimé… » (pour éviter les faux positifs).
- Sans colonne `aliases`, les titres français ne sont pas reconnus dans le mode règles (le LLM, lui, les traduit en général).
- Identifiants du catalogue mock fictifs : ne pas les mélanger avec ceux de MovieLens.
