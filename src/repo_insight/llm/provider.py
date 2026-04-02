"""LLM provider factory — wraps langchain-openai ChatOpenAI for any OpenAI-compatible endpoint.

This module is the single point of construction for LLM client objects used
throughout the application (agent nodes, report generation, etc.).  By
centralizing creation here, every caller automatically picks up the base URL,
model name, and API key from the application configuration without having to
import ``settings`` themselves.

The factory function returns a ``ChatOpenAI`` instance from the
``langchain-openai`` package.  Despite the name, ``ChatOpenAI`` supports any
HTTP endpoint that implements the OpenAI chat-completions API contract — this
includes Ollama, DeepSeek, Azure OpenAI, LiteLLM proxies, and others.

Reasoning / Thinking Support
-----------------------------
Many reasoning-capable LLMs (e.g. Ollama qwen3 with ``thinking`` capability,
DeepSeek-R1) return a "thinking" / "reasoning" field alongside the regular
``content`` field in their streaming SSE deltas.  However, langchain-openai's
internal ``_convert_delta_to_message_chunk`` function only extracts ``content``,
``tool_calls``, and ``function_call`` — any extra vendor-specific fields such
as ``reasoning`` (Ollama) or ``reasoning_content`` (DeepSeek) are silently
discarded.

To surface reasoning tokens to the rest of the application **without modifying
langchain source code**, this module provides two approaches:

**Approach 1 — Monkey-patch** (DEFAULT, enabled at module load time):
  Wraps the original ``_convert_delta_to_message_chunk`` with a thin decorator
  that copies the ``reasoning`` or ``reasoning_content`` field from the raw
  delta dict into the resulting ``AIMessageChunk.additional_kwargs``.  This is
  the simplest solution (~15 lines) and works transparently for all callers.

  *Risk*: ``_convert_delta_to_message_chunk`` is a private function (prefixed
  with ``_``) in ``langchain_openai.chat_models.base``.  Its signature or
  behavior may change across langchain-openai releases.  If an upgrade breaks
  the patch, a ``WARNING`` log is emitted and the original function is left
  untouched — the application keeps working, but reasoning tokens are lost.

**Approach 2 — ChatOpenAI subclass** (AVAILABLE, not used by default):
  ``ChatOpenAIWithReasoning`` inherits from ``ChatOpenAI`` and overrides
  ``_convert_chunk_to_generation_chunk`` to inject reasoning data from the raw
  API chunk into the message's ``additional_kwargs``.  This is more "proper" in
  OOP terms and survives internal refactors of the delta conversion path, but
  requires changing ``get_llm()`` to return the subclass.

  To switch: change ``get_llm`` to return ``ChatOpenAIWithReasoning(...)``
  instead of ``ChatOpenAI(...)``.

Both approaches are safe for models that do *not* produce reasoning output —
the extra field simply won't exist in the delta, so the patch / override is a
no-op.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from repo_insight.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Approach 1 — Monkey-patch  (ACTIVE by default)
# =============================================================================
# Why monkey-patch?
#   langchain-openai's ``_convert_delta_to_message_chunk`` is the single
#   bottleneck that converts raw OpenAI-format SSE delta dicts into LangChain
#   ``BaseMessageChunk`` objects.  It deliberately ignores unknown fields, so
#   vendor extensions like ``reasoning`` (Ollama for qwen3 thinking models) and
#   ``reasoning_content`` (DeepSeek-R1) are dropped before any callback or
#   streaming handler ever sees them.
#
# How it works:
#   We wrap the original function: after it produces the message chunk we
#   inspect the raw delta dict for reasoning fields and, if found, stash them
#   in ``chunk.additional_kwargs["reasoning"]``.  Downstream code (e.g. the SSE
#   event generator in ``chat.py``) can then read this key to forward reasoning
#   tokens to the frontend.
#
# Compatibility:
#   - Ollama qwen3 (thinking models)  → delta field name: "reasoning"
#   - DeepSeek-R1 (official API)      → delta field name: "reasoning_content"
#   - OpenAI o-series (o1, o3, o4)    → uses Responses API, separate code path
#                                        in langchain; this patch is irrelevant
#   - Models without thinking          → no extra field, patch is a harmless no-op
#
# Risk:
#   ``_convert_delta_to_message_chunk`` is a private function.  If langchain-openai
#   renames it, changes its signature, or refactors the streaming pipeline, this
#   patch will fail.  The try/except ensures the application still works — only
#   reasoning passthrough is lost.  A WARNING log alerts operators.
# -----------------------------------------------------------------------------

# Field names that different providers use for reasoning/thinking content
# in their SSE delta objects.  Add new vendor field names here as needed.
_REASONING_FIELD_NAMES = ("reasoning", "reasoning_content")


def _apply_reasoning_monkey_patch() -> None:
    """Apply the monkey-patch to ``_convert_delta_to_message_chunk``.

    Called once at module load time.  Wrapped in a function to keep the
    module namespace clean and make the patch easy to disable by commenting
    out a single call.
    """
    try:
        import langchain_openai.chat_models.base as _base

        _original_fn = _base._convert_delta_to_message_chunk

        def _patched_convert_delta(
            _dict, default_class
        ):
            """Wrapper that preserves reasoning tokens from the raw delta.

            Calls the original ``_convert_delta_to_message_chunk`` to produce
            a properly typed ``BaseMessageChunk``, then checks the raw delta
            dict for any known reasoning field names.  If found, the reasoning
            text is injected into ``chunk.additional_kwargs["reasoning"]`` so
            that it is accessible to downstream streaming handlers.

            Parameters
            ----------
            _dict : Mapping[str, Any]
                The raw delta dictionary from the SSE chunk (e.g.
                ``{"role": "assistant", "content": "...", "reasoning": "..."}``)
            default_class : type[BaseMessageChunk]
                The fallback message chunk class when role is not specified.

            Returns
            -------
            BaseMessageChunk
                The original chunk, potentially augmented with reasoning data.
            """
            chunk = _original_fn(_dict, default_class)

            # Look for reasoning content under each known field name.
            # Different providers use different keys:
            #   - Ollama (qwen3 thinking): "reasoning"
            #   - DeepSeek-R1:             "reasoning_content"
            for field_name in _REASONING_FIELD_NAMES:
                reasoning_text = _dict.get(field_name)
                if reasoning_text:
                    # Store in additional_kwargs under the canonical key
                    # "reasoning" regardless of the provider's field name,
                    # so downstream consumers have a single key to check.
                    chunk.additional_kwargs["reasoning"] = reasoning_text
                    break  # first match wins; avoid overwriting

            return chunk

        _base._convert_delta_to_message_chunk = _patched_convert_delta
        logger.info(
            "Reasoning monkey-patch applied to "
            "langchain_openai._convert_delta_to_message_chunk"
        )

    except Exception as e:
        # If the patch fails (e.g. langchain-openai refactored the function
        # away), log a warning but do NOT crash.  The application continues
        # working normally — only reasoning passthrough is lost.
        logger.warning(
            "Failed to apply reasoning monkey-patch — reasoning tokens will "
            "not be forwarded to the frontend.  This is non-fatal.  "
            "Error: %s",
            e,
        )


# Apply the patch at module load time.  Every import of this module
# (which happens very early, since agents and routes import get_llm)
# ensures the patch is in place before any LLM call occurs.
_apply_reasoning_monkey_patch()


# =============================================================================
# Approach 2 — ChatOpenAI subclass  (AVAILABLE, not used by default)
# =============================================================================
# Why a subclass?
#   Overriding ``_convert_chunk_to_generation_chunk`` is the officially intended
#   extension point in ``BaseChatOpenAI``.  It receives the full raw API chunk
#   dict (not just the delta), so it can extract reasoning data at a higher
#   level than the monkey-patch.
#
# Why not used by default?
#   The monkey-patch (Approach 1) is simpler and already handles the common
#   case.  This subclass is provided as a more robust alternative if:
#     - The monkey-patch breaks after a langchain-openai upgrade.
#     - You prefer an OOP approach over patching private functions.
#     - You need per-instance control (e.g. only some LLM instances should
#       capture reasoning).
#
# How to activate:
#   Change the ``return`` statement in ``get_llm()`` below to use
#   ``ChatOpenAIWithReasoning(...)`` instead of ``ChatOpenAI(...)``.
#
# Risk:
#   ``_convert_chunk_to_generation_chunk`` is also technically a private method,
#   but it is explicitly designed as a hook in ``BaseChatOpenAI`` and is less
#   likely to change without notice.  If it does change, Python will raise a
#   clear ``TypeError`` at call time.
# -----------------------------------------------------------------------------

class ChatOpenAIWithReasoning(ChatOpenAI):
    """ChatOpenAI subclass that preserves reasoning/thinking tokens.

    Overrides ``_convert_chunk_to_generation_chunk`` to inspect the raw
    API streaming chunk for reasoning data before delegating to the parent
    implementation.  Any reasoning text found is stored in
    ``generation_chunk.message.additional_kwargs["reasoning"]``.

    This class is **not used by default** — see module docstring for details
    on how to activate it.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ):
        """Override: extract reasoning from the raw API chunk, then delegate.

        The raw ``chunk`` dict mirrors the full SSE object from the provider
        (e.g. ``{"choices": [{"delta": {"content": "...", "reasoning": "..."}}]}``).
        We extract reasoning from the delta *before* calling ``super()`` because
        the parent method only reads ``content`` and ``tool_calls``.

        Parameters
        ----------
        chunk : dict
            Raw streaming chunk from the OpenAI-compatible API.
        default_chunk_class : type
            Default message chunk class (usually ``AIMessageChunk``).
        base_generation_info : dict or None
            Base metadata to attach to the generation.

        Returns
        -------
        ChatGenerationChunk or None
            The generation chunk with reasoning injected into
            ``message.additional_kwargs``, or None if the parent returns None.
        """
        gen_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )

        if gen_chunk is None:
            return None

        # Extract reasoning from the delta inside the first choice.
        # The structure is: {"choices": [{"delta": {"reasoning": "..."}}]}
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            for field_name in _REASONING_FIELD_NAMES:
                reasoning_text = delta.get(field_name)
                if reasoning_text:
                    gen_chunk.message.additional_kwargs["reasoning"] = reasoning_text
                    break

        return gen_chunk


# =============================================================================
# Factory function
# =============================================================================

def get_llm(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    streaming: bool = True,
) -> ChatOpenAI:
    """Create and return a ``ChatOpenAI`` instance pointed at the configured LLM endpoint.

    This is the **factory function** used by every part of the application
    that needs to talk to an LLM.  Callers can override individual parameters
    for special cases (e.g. a lower temperature for deterministic report
    formatting), but the defaults are designed for general conversational use.

    Supports Ollama, DeepSeek, OpenAI, or any OpenAI-compatible provider.

    By default this returns a standard ``ChatOpenAI`` instance (Approach 1's
    monkey-patch handles reasoning passthrough globally).  To use Approach 2
    instead, replace ``ChatOpenAI`` with ``ChatOpenAIWithReasoning`` below.

    Parameters
    ----------
    base_url : str or None
        The root URL of the LLM API (e.g. ``"https://api.openai.com/v1"``).
        When ``None`` (the default), the value from ``settings.llm_base_url``
        is used, which originates from the ``LLM_BASE_URL`` env var.
    model : str or None
        The model identifier to request (e.g. ``"gpt-4o"``, ``"deepseek-chat"``).
        Defaults to ``settings.llm_model`` (``LLM_MODEL`` env var).
    api_key : str or None
        The bearer-token API key for authentication.  Defaults to
        ``settings.llm_api_key`` (``LLM_API_KEY`` env var).
    temperature : float
        Sampling temperature controlling response randomness.  Lower values
        (e.g. 0.0) produce more deterministic output; higher values (e.g. 1.0)
        increase creativity.  The default of 0.7 balances informativeness with
        variety for conversational interactions.
    streaming : bool
        Whether to request token-level streaming from the provider.  Must be
        ``True`` for the CLI's progressive-display feature to work; set to
        ``False`` when only the final, complete response is needed.

    Returns
    -------
    ChatOpenAI
        A fully configured LangChain chat-model instance ready for
        ``.invoke()``, ``.ainvoke()``, or ``.astream()`` calls.
    """
    # DEFAULT: Uses standard ChatOpenAI + Approach 1 monkey-patch.
    # To switch to Approach 2, change ChatOpenAI → ChatOpenAIWithReasoning.
    return ChatOpenAI(
        base_url=base_url or settings.llm_base_url,
        model=model or settings.llm_model,
        api_key=api_key or settings.llm_api_key,
        temperature=temperature,
        streaming=streaming,
    )
