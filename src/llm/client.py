"""Petit client LLM sans dépendance (urllib), avec repli hors-ligne.

Choix du fournisseur (variables d'environnement) :
  LLM_OFFLINE=1                 -> jamais d'appel réseau (mode prof sans clé)
  ANTHROPIC_API_KEY             -> API Anthropic (LLM_MODEL, défaut claude-haiku-5-5)
  OPENAI_API_KEY (+ OPENAI_BASE_URL) -> toute API compatible OpenAI : OpenAI, Groq, Mistral,
                                    Ollama local (OPENAI_BASE_URL=http://localhost:11434/v1)
  LLM_PROVIDER=anthropic|openai -> force un fournisseur si les deux clés sont présentes

complete() renvoie None en cas d'absence de clé ou d'erreur : l'appelant bascule alors
sur le mode règles + phrases toutes faites.
"""
from __future__ import annotations

import json
import os
import urllib.request

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "25"))


def provider() -> str | None:
    if os.getenv("LLM_OFFLINE") == "1":
        return None
    forced = os.getenv("LLM_PROVIDER")
    if forced in ("anthropic", "openai"):
        return forced
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        return "openai"
    return None


def is_available() -> bool:
    return provider() is not None


def _post(url: str, headers: dict, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def complete(system: str, messages: list[dict], max_tokens: int = 800, temperature: float = 0.3,
             json_mode: bool = False) -> str | None:
    """messages : [{"role": "user"|"assistant", "content": str}]"""
    p = provider()
    if p is None:
        return None
    try:
        if p == "anthropic":
            data = _post(
                "https://api.anthropic.com/v1/messages",
                {
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                {
                    "model": os.getenv("LLM_MODEL", "claude-haiku-5-5"),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system,
                    "messages": messages,
                },
            )
            return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        body = {
            "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = _post(
            f"{base}/chat/completions",
            {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'ollama')}", "Content-Type": "application/json"},
            body,
        )
        return data["choices"][0]["message"]["content"]
    except Exception as e:  # réseau, quota, clé invalide… -> repli hors-ligne
        if os.getenv("LLM_DEBUG") == "1":
            print(f"[llm] erreur {p}: {e!r}")
        return None


def parse_json(text: str | None) -> dict | None:
    """Extrait le premier objet JSON d'une réponse (tolère ```json … ```)."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
