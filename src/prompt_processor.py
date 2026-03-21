"""
Prompt Processor for SDXL Interior Inpainting
==============================================
Converts vague, human-written prompts into structured positive + negative
prompt pairs optimised for SDXL inpainting of interior spaces.

Three inpainting task types are supported:
  • ADD    – place a new object / furniture into the scene
  • REMOVE – erase an object and fill naturally with the surrounding interior
  • EDIT   – modify an existing object (colour, style, material, etc.)

LLM backends (tried in order, first available wins):
  1. OpenAI-compatible API  (GPT-4o-mini or any model via OPENAI_API_KEY /
                              OPENAI_BASE_URL env vars, or explicit constructor args)
  2. HuggingFace Transformers pipeline (local model, e.g. Qwen/Qwen2.5-1.5B-Instruct)
  3. Ollama  (local server, default model "qwen2.5:1.5b")
  4. Rule-based fallback  (no LLM needed – deterministic quality boost)

Usage
-----
from src.prompt_processor import PromptProcessor, TaskType

# Auto-detect backend
pp = PromptProcessor()

result = pp.process("add a blue velvet sofa")
print(result.task_type)          # TaskType.ADD
print(result.positive_prompt)
print(result.negative_prompt)

# Force a specific backend
pp = PromptProcessor(backend="openai", model="gpt-4o-mini", api_key="sk-...")
pp = PromptProcessor(backend="hf",     model="Qwen/Qwen2.5-1.5B-Instruct")
pp = PromptProcessor(backend="ollama", model="qwen2.5:1.5b")
pp = PromptProcessor(backend="rule")   # no LLM, always available
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    ADD    = "add"
    REMOVE = "remove"
    EDIT   = "edit"
    UNKNOWN = "unknown"


@dataclass
class PromptResult:
    raw_prompt: str
    task_type: TaskType
    positive_prompt: str
    negative_prompt: str
    backend_used: str
    meta: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.task_type.value.upper()}] (via {self.backend_used})\n"
            f"  + {self.positive_prompt}\n"
            f"  - {self.negative_prompt}"
        )


# ---------------------------------------------------------------------------
# Task-type keyword detection
# ---------------------------------------------------------------------------

_ADD_KEYWORDS = (
    r"\b(add|place|put|insert|include|bring|introduce|hang|install|lay|set|"
    r"thêm|đặt|đưa vào|gắn|lắp|treo)\b"
)
_REMOVE_KEYWORDS = (
    r"\b(remove|delete|erase|eliminate|take out|get rid of|clear|clean up|"
    r"xóa|bỏ|dọn|loại bỏ|xoá)\b"
)
_EDIT_KEYWORDS = (
    r"\b(change|replace|modify|update|repaint|recolor|recolour|swap|turn|make|"
    r"style|redesign|renovate|transform|convert|thay|đổi|sửa|sơn|chỉnh)\b"
)


def detect_task_type(prompt: str) -> TaskType:
    """Heuristic keyword-based task-type detection (case-insensitive)."""
    p = prompt.lower()
    if re.search(_ADD_KEYWORDS, p):
        return TaskType.ADD
    if re.search(_REMOVE_KEYWORDS, p):
        return TaskType.REMOVE
    if re.search(_EDIT_KEYWORDS, p):
        return TaskType.EDIT
    return TaskType.UNKNOWN


# ---------------------------------------------------------------------------
# System prompts per task type
# ---------------------------------------------------------------------------

_SYSTEM_COMMON = (
    "You are an expert prompt engineer specialising in Stable Diffusion XL "
    "inpainting for interior photography. "
    "Given a short, imprecise human instruction about an interior scene, "
    "your job is to produce TWO prompt strings that will be fed directly to "
    "an SDXL inpainting diffusion model:\n"
    "  1. positive_prompt – describes WHAT should appear in the inpainted region, "
    "written in the concise, comma-separated tag style that works best for SDXL. "
    "Always emphasise photorealism, proper lighting, and seamless blending with "
    "the rest of the room.\n"
    "  2. negative_prompt – lists artefacts, distortions, and stylistic elements "
    "to AVOID, e.g. cartoon, blurry, low quality, watermark, floating objects, "
    "wrong perspective, mismatched lighting.\n\n"
    "Return ONLY a JSON object with exactly two keys: "
    '"positive_prompt" and "negative_prompt". No prose, no markdown fences.'
)

_TASK_ADDENDUM: dict[TaskType, str] = {
    TaskType.ADD: (
        "TASK: ADD a new object into an existing interior space.\n"
        "Tips for positive_prompt: specify the object, its style (modern/Scandinavian/"
        "industrial/etc.), material, colour, and how it fits the surrounding decor. "
        "Include anchor phrases like 'naturally placed', 'physically grounded', "
        "'matching ambient lighting', 'seamless integration'.\n"
        "Tips for negative_prompt: floating, levitating, misplaced, wrong scale, "
        "clipping, shadow mismatch, cartoon, anime, illustration, blurry, low quality, "
        "watermark, nsfw."
    ),
    TaskType.REMOVE: (
        "TASK: REMOVE an object and fill the empty area to match the surrounding interior.\n"
        "Tips for positive_prompt: describe what the gap should look like after removal – "
        "floor, wall, or background surface, matching the existing texture/colour/material. "
        "Include 'seamless background completion', 'consistent floor/wall texture', "
        "'realistic continuation', 'no visible editing artifacts'.\n"
        "Tips for negative_prompt: object residue, ghost shadow, visible seam, incomplete fill, "
        "mismatched texture, colour inconsistency, blurry, low quality, watermark."
    ),
    TaskType.EDIT: (
        "TASK: EDIT / MODIFY an existing object in the interior (colour, style, material, etc.).\n"
        "Tips for positive_prompt: describe the target appearance of the modified object, "
        "keeping geometry consistent. Mention 'same shape and position', 'updated material/colour', "
        "'photorealistic texture', 'matching room lighting'.\n"
        "Tips for negative_prompt: shape change, deformation, wrong pose, cartoon, blurry, "
        "over-saturated, watermark, low quality, mismatched scale."
    ),
    TaskType.UNKNOWN: (
        "TASK TYPE is unclear – treat it as a general interior inpainting edit.\n"
        "Apply your best judgment to produce high-quality positive and negative prompts."
    ),
}


def _build_system_prompt(task_type: TaskType) -> str:
    return _SYSTEM_COMMON + "\n\n" + _TASK_ADDENDUM[task_type]


# ---------------------------------------------------------------------------
# Rule-based fallback (no LLM)
# ---------------------------------------------------------------------------

_QUALITY_TAGS = (
    "8k uhd, hyperrealistic interior photography, professional lighting, "
    "sharp focus, canon eos r5, 24mm lens, architectural digest style"
)

_BASE_NEGATIVE = (
    "cartoon, anime, illustration, painting, sketch, 3d render, blurry, "
    "low quality, low resolution, watermark, text, signature, noise, grain, "
    "oversaturated, ugly, deformed, bad anatomy, duplicate, extra objects, "
    "floating objects, levitating furniture, shadow mismatch, mismatched lighting, "
    "wrong perspective, distortion, artefacts, jpeg artefacts"
)

_TASK_NEGATIVE_EXTRA: dict[TaskType, str] = {
    TaskType.ADD: (
        "missing, invisible, partially visible, clipping through floor, "
        "wrong scale, misplaced"
    ),
    TaskType.REMOVE: (
        "object residue, ghost, shadow remnant, incomplete fill, "
        "visible seam, colour inconsistency, mismatched texture"
    ),
    TaskType.EDIT: (
        "shape deformation, pose change, wrong geometry, flat color, "
        "plastic look, over-processed"
    ),
    TaskType.UNKNOWN: "",
}

_TASK_POSITIVE_PREFIX: dict[TaskType, str] = {
    TaskType.ADD: (
        "photorealistic interior, naturally placed, physically grounded, "
        "matching ambient lighting, seamless integration, "
    ),
    TaskType.REMOVE: (
        "seamless background completion, consistent floor texture, "
        "consistent wall texture, realistic continuation, no editing artefacts, "
    ),
    TaskType.EDIT: (
        "same shape and position, updated material, photorealistic texture, "
        "matching room lighting, consistent perspective, "
    ),
    TaskType.UNKNOWN: "photorealistic interior design, high quality, ",
}


def _rule_based_enhance(raw_prompt: str, task_type: TaskType) -> PromptResult:
    """Deterministic rule-based prompt enhancement – no LLM required."""
    prefix = _TASK_POSITIVE_PREFIX[task_type]
    positive = f"{prefix}{raw_prompt.strip().rstrip(',')}, {_QUALITY_TAGS}"

    neg_extra = _TASK_NEGATIVE_EXTRA[task_type]
    negative = _BASE_NEGATIVE + (f", {neg_extra}" if neg_extra else "")

    return PromptResult(
        raw_prompt=raw_prompt,
        task_type=task_type,
        positive_prompt=positive,
        negative_prompt=negative,
        backend_used="rule",
    )


# ---------------------------------------------------------------------------
# LLM helper utilities
# ---------------------------------------------------------------------------

def _parse_llm_json(text: str) -> tuple[str, str]:
    """Extract positive/negative prompts from LLM JSON output (lenient parser)."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

    data = json.loads(text)
    positive = str(data.get("positive_prompt") or data.get("positive") or "").strip()
    negative = str(data.get("negative_prompt") or data.get("negative") or "").strip()

    if not positive:
        raise ValueError("LLM response missing 'positive_prompt'")
    return positive, negative


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class _OpenAIBackend:
    name = "openai"

    def __init__(self, model: str, api_key: str | None, base_url: str | None) -> None:
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.model = model or "gpt-4o-mini"

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        return _parse_llm_json(raw)


class _HFBackend:
    name = "hf"

    def __init__(self, model: str) -> None:
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers package not installed. Run: pip install transformers"
            ) from exc

        self.model_name = model or "Qwen/Qwen2.5-1.5B-Instruct"
        logger.info("Loading HuggingFace model '%s' …", self.model_name)
        self._pipe = hf_pipeline(
            "text-generation",
            model=self.model_name,
            device_map="auto",
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        result = self._pipe(messages)
        # Transformers pipeline returns list of dicts
        generated = result[0]["generated_text"]
        # The last entry in generated_text is the assistant's reply
        if isinstance(generated, list):
            raw = generated[-1].get("content", "")
        else:
            raw = str(generated)
        return _parse_llm_json(raw)


class _OllamaBackend:
    name = "ollama"

    def __init__(self, model: str, base_url: str | None) -> None:
        try:
            import requests  # type: ignore
            self._requests = requests
        except ImportError as exc:
            raise ImportError(
                "requests package not installed. Run: pip install requests"
            ) from exc

        self.model = model or "qwen2.5:1.5b"
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def _is_available(self) -> bool:
        try:
            r = self._requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.3},
            "format": "json",
        }
        r = self._requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["message"]["content"]
        return _parse_llm_json(raw)


# ---------------------------------------------------------------------------
# Main PromptProcessor class
# ---------------------------------------------------------------------------

_BACKEND_ORDER = ("openai", "hf", "ollama", "rule")


class PromptProcessor:
    """
    Converts a raw human prompt into a (positive_prompt, negative_prompt) pair
    optimised for SDXL inpainting of interior spaces.

    Parameters
    ----------
    backend : str | None
        One of "openai", "hf", "ollama", "rule", or None (auto-detect).
    model : str | None
        Model name for the chosen backend.
    api_key : str | None
        API key (OpenAI backend).  Falls back to OPENAI_API_KEY env var.
    base_url : str | None
        Custom API base URL (OpenAI-compatible servers or Ollama).
    auto_detect_task : bool
        If True (default) infer task type from the prompt; otherwise TaskType.UNKNOWN.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        auto_detect_task: bool = True,
    ) -> None:
        self._auto_detect_task = auto_detect_task
        self._backend_obj = self._init_backend(backend, model, api_key, base_url)
        logger.info("PromptProcessor using backend: %s", self._backend_obj.name
                    if hasattr(self._backend_obj, "name") else "rule")

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_backend(
        self,
        backend: str | None,
        model: str | None,
        api_key: str | None,
        base_url: str | None,
    ):
        if backend == "rule":
            return _RuleBackend()

        if backend == "openai" or (backend is None and self._has_openai_key(api_key)):
            try:
                return _OpenAIBackend(model or "gpt-4o-mini", api_key, base_url)
            except Exception as e:
                if backend == "openai":
                    raise
                logger.warning("OpenAI backend unavailable: %s", e)

        if backend == "hf":
            return _HFBackend(model or "Qwen/Qwen2.5-1.5B-Instruct")

        if backend == "ollama" or backend is None:
            try:
                obj = _OllamaBackend(model or "qwen2.5:1.5b", base_url)
                if obj._is_available():
                    return obj
                if backend == "ollama":
                    raise RuntimeError(
                        f"Ollama server not reachable at {obj.base_url}. "
                        "Start with: ollama serve"
                    )
            except RuntimeError:
                raise
            except Exception as e:
                if backend == "ollama":
                    raise
                logger.warning("Ollama backend unavailable: %s", e)

        # Auto-detect fell through → rule-based
        logger.info(
            "No LLM backend available; using rule-based prompt enhancement."
        )
        return _RuleBackend()

    @staticmethod
    def _has_openai_key(explicit_key: str | None) -> bool:
        return bool(explicit_key or os.environ.get("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        prompt: str,
        task_type: Optional[TaskType] = None,
    ) -> PromptResult:
        """
        Process a raw human prompt.

        Parameters
        ----------
        prompt : str
            The raw instruction, e.g. "add a blue velvet sofa on the left side".
        task_type : TaskType | None
            Override automatic task detection.

        Returns
        -------
        PromptResult
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        if task_type is None:
            task_type = detect_task_type(prompt) if self._auto_detect_task else TaskType.UNKNOWN

        # Rule-based path (fast, no LLM call)
        if isinstance(self._backend_obj, _RuleBackend):
            return _rule_based_enhance(prompt, task_type)

        # LLM path
        system_prompt = _build_system_prompt(task_type)
        user_message = (
            f'Interior inpainting instruction: "{prompt}"\n\n'
            f"Detected task type: {task_type.value.upper()}\n\n"
            "Return a JSON object with keys 'positive_prompt' and 'negative_prompt'."
        )

        try:
            positive, negative = self._backend_obj.generate(system_prompt, user_message)
        except Exception as exc:
            logger.warning(
                "LLM backend '%s' failed (%s). Falling back to rule-based enhancement.",
                getattr(self._backend_obj, "name", "unknown"),
                exc,
            )
            return _rule_based_enhance(prompt, task_type)

        return PromptResult(
            raw_prompt=prompt,
            task_type=task_type,
            positive_prompt=positive,
            negative_prompt=negative,
            backend_used=getattr(self._backend_obj, "name", "llm"),
        )

    def process_batch(
        self,
        prompts: list[str],
        task_types: Optional[list[Optional[TaskType]]] = None,
    ) -> list[PromptResult]:
        """Process multiple prompts. task_types list is optional (auto-detect)."""
        if task_types is None:
            task_types = [None] * len(prompts)
        return [self.process(p, t) for p, t in zip(prompts, task_types)]


# ---------------------------------------------------------------------------
# Thin wrapper to give _RuleBackend the same duck-type interface
# ---------------------------------------------------------------------------

class _RuleBackend:
    name = "rule"

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:  # pragma: no cover
        # Never called directly – PromptProcessor short-circuits to _rule_based_enhance
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Convenience function (module-level shortcut)
# ---------------------------------------------------------------------------

_default_processor: Optional[PromptProcessor] = None


def process_prompt(
    prompt: str,
    task_type: Optional[TaskType] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> PromptResult:
    """
    Module-level convenience wrapper around PromptProcessor.

    Uses a cached default processor (auto-detect backend).
    Pass explicit `backend`/`model`/`api_key` to override.

    Examples
    --------
    >>> from src.prompt_processor import process_prompt
    >>> result = process_prompt("remove the old carpet")
    >>> print(result.positive_prompt)
    """
    global _default_processor
    if backend or model or api_key or base_url or _default_processor is None:
        _default_processor = PromptProcessor(
            backend=backend, model=model, api_key=api_key, base_url=base_url
        )
    return _default_processor.process(prompt, task_type)
