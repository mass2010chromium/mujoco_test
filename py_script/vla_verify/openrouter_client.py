"""
OpenRouter API client for querying LLMs and VLMs.

Supports both text-only (LLM) and multimodal (VLM: image + text) queries
through the OpenRouter unified API. Used by both the scene graph constructor
(VLM) and the subtask translator (LLM).
"""

import base64
import json
import os
from pathlib import Path
from typing import Optional

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_VLM_MODEL = "google/gemini-2.5-pro"
DEFAULT_LLM_MODEL = "google/gemini-2.5-flash"


def _get_api_key(api_key: Optional[str] = None) -> str:
    """Resolve the OpenRouter API key from argument or environment."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Export it or pass api_key=..."
        )
    return key.strip()


def _encode_image(image_path: Path) -> tuple:
    """Encode an image file to base64 and determine its MIME type."""
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/png")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime


def query_vlm(
    image_path: Path,
    prompt: str,
    model: str = DEFAULT_VLM_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    """
    Query a Vision-Language Model with an image and text prompt via OpenRouter.

    Args:
        image_path: Path to the image file.
        prompt: Text prompt to send with the image.
        model: OpenRouter model identifier (must support vision).
        api_key: API key (falls back to OPENROUTER_API_KEY env var).
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.

    Returns:
        The model's text response.
    """
    key = _get_api_key(api_key)
    base64_data, mime = _encode_image(image_path)
    data_url = f"data:{mime};base64,{base64_data}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices")
    if not choices:
        raise RuntimeError(f"Unexpected VLM response: {data}")
    return choices[0].get("message", {}).get("content", "").strip()


def query_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    timeout: int = 120,
) -> str:
    """
    Query a text-only Language Model via OpenRouter.

    Args:
        prompt: User prompt text.
        system_prompt: Optional system/instruction prompt.
        model: OpenRouter model identifier.
        api_key: API key (falls back to OPENROUTER_API_KEY env var).
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.

    Returns:
        The model's text response.
    """
    key = _get_api_key(api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices")
    if not choices:
        raise RuntimeError(f"Unexpected LLM response: {data}")
    return choices[0].get("message", {}).get("content", "").strip()
