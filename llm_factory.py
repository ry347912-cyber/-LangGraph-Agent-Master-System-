"""
utils/llm_factory.py — LLM Provider Abstraction
Switch between OpenAI, Anthropic, Gemini with one line.
Includes token cost estimation.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Cost per 1K tokens (input, output) in USD
COST_TABLE = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
}


def get_llm(
    provider: str = None,
    model: str = None,
    temperature: float = 0.1,
    streaming: bool = False
):
    """
    Get LLM instance. Provider auto-detected from env if not specified.
    
    Usage:
        llm = get_llm()                          # Uses defaults
        llm = get_llm("anthropic", temperature=0) # Force Anthropic
        llm = get_llm(streaming=True)            # For streaming endpoints
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.debug(f"LLM: OpenAI {model}")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            max_tokens=2000
        )

    elif provider == "anthropic":
        model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        logger.debug(f"LLM: Anthropic {model}")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            streaming=streaming,
            max_tokens=2000
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'")


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given model and token counts."""
    if model not in COST_TABLE:
        return 0.0
    input_rate, output_rate = COST_TABLE[model]
    return (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)
