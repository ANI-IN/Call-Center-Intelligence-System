"""LLM provider factory supporting multiple backends.

Supported providers:
- openai: GPT-4o (default, paid)
- gemini: Gemini 2.0 Flash (free tier: 15 RPM, 1500 RPD)
- groq: Llama 3.3 70B (free tier: 30 RPM, 15K tokens/min)
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
}


def get_llm(
    provider: str = "openai",
    model: str | None = None,
    timeout: int = 120,
) -> BaseChatModel:
    """Create an LLM client for the given provider.

    Args:
        provider: One of "openai", "gemini", "groq"
        model: Model name override (uses provider default if None)
        timeout: Request timeout in seconds
    """
    model_name = model or DEFAULT_MODELS.get(provider, "gpt-4o")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, timeout=timeout)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, timeout=timeout)

    # Default: OpenAI
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, timeout=timeout)
