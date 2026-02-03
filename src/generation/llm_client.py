from __future__ import annotations
import os
import requests
from typing import List, Dict, Optional


class LLMError(Exception):
    pass


def chat_completion(
    messages: List[Dict],
    model: str,
    provider: str = "groq",
    api_key: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> str:
    provider = provider.lower().strip()
    if provider == "groq":
        return chat_completion_groq(
            messages=messages,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif provider == "openrouter":
        return chat_completion_openrouter(
            messages=messages,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise LLMError(f"Unknown provider: {provider}")


def chat_completion_groq(
    messages: List[Dict],
    model: str,
    api_key: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> str:
    """
    Groq OpenAI-compatible chat completions endpoint.
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMError("Missing GROQ_API_KEY environment variable.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise LLMError(f"Groq error {r.status_code}: {r.text}")

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise LLMError(f"Unexpected response format: {data}") from e


def chat_completion_openrouter(
    messages: List[Dict],
    model: str,
    api_key: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> str:
    """
    OpenRouter chat completions endpoint.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("Missing OPENROUTER_API_KEY environment variable.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "patient-rag",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise LLMError(f"OpenRouter error {r.status_code}: {r.text}")

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise LLMError(f"Unexpected response format: {data}") from e
