"""One text-completion interface over several model vendors.

The frame probe needs to ask the same question of models from different
providers, so the vendor differences are isolated here and nothing else in the
repo knows about them. Model names are vendor-prefixed:

    anthropic:claude-opus-5        openai:gpt-5        google:gemini-2.5-pro

A bare name defaults to `anthropic:`.

Two design points that matter for the experiment rather than for tidiness:

* **Identical prompts.** Every vendor gets the same system prompt and the same
  user text, byte for byte. Vendor-specific prompt tuning would make the
  comparison meaningless -- the point is what a model does with the sentence, not
  what it does after I have optimised around it.
* **Deterministic settings.** Temperature 0 where the API allows it. A probe
  measuring whether a model's answer *changes* between two phrasings must not
  have sampling noise as an alternative explanation.

`preflight()` reports which models are actually reachable before anything is
spent, because a half-finished sweep across three vendors is worse than none.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DEFAULT_VENDOR = "anthropic"

#: Environment variable each vendor's client needs.
KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
}

#: Suggested current frontier models per vendor, for `--model frontier`. These
#: are names, not endorsements, and they go stale -- pass explicit names to pin.
FRONTIER = (
    "anthropic:claude-opus-5",
    "openai:gpt-5",
    "google:gemini-2.5-pro",
)


def split_name(name: str) -> Tuple[str, str]:
    if ":" in name:
        vendor, model = name.split(":", 1)
        return vendor.strip().lower(), model.strip()
    return DEFAULT_VENDOR, name.strip()


def key_present(vendor: str) -> bool:
    env = KEY_ENV.get(vendor)
    return bool(env and os.environ.get(env))


class VendorClient:
    """Lazily-constructed client with a uniform `complete()`."""

    def __init__(self, name: str, max_tokens: int = 300,
                 temperature: float = 0.0):
        self.vendor, self.model = split_name(name)
        self.name = f"{self.vendor}:{self.model}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        if self.vendor not in KEY_ENV:
            raise ValueError(
                f"unknown vendor {self.vendor!r}; have {sorted(KEY_ENV)}")

    def _ensure(self):
        if self._client is not None:
            return
        env = KEY_ENV[self.vendor]
        if not os.environ.get(env):
            raise RuntimeError(f"{env} is not set, so {self.name} cannot run")
        if self.vendor == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        elif self.vendor == "openai":
            import openai
            self._client = openai.OpenAI()
        elif self.vendor == "google":
            from google import genai
            self._client = genai.Client(api_key=os.environ[env])

    def complete(self, system: str, user: str) -> str:
        """Return the model's text. Same prompt for every vendor."""
        self._ensure()
        if self.vendor == "anthropic":
            msg = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                temperature=self.temperature, system=system,
                messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in msg.content
                           if getattr(b, "type", "") == "text")
        if self.vendor == "openai":
            # Some reasoning models reject temperature; retry without it rather
            # than silently skipping the model.
            kw = dict(model=self.model,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}])
            try:
                r = self._client.chat.completions.create(
                    **kw, max_completion_tokens=self.max_tokens,
                    temperature=self.temperature)
            except Exception:
                r = self._client.chat.completions.create(
                    **kw, max_completion_tokens=self.max_tokens)
            return r.choices[0].message.content or ""
        if self.vendor == "google":
            from google.genai import types
            r = self._client.models.generate_content(
                model=self.model, contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature))
            return r.text or ""
        raise AssertionError(self.vendor)


def preflight(names: List[str], verbose: bool = True) -> Dict[str, Dict]:
    """Check each model is reachable with a one-token question.

    Run before a sweep. A partially-completed comparison across vendors is worse
    than no comparison, because the missing arm is exactly the one a reader will
    assume was omitted for being inconvenient.
    """
    out: Dict[str, Dict] = {}
    for name in names:
        vendor, model = split_name(name)
        row: Dict = {"vendor": vendor, "model": model,
                     "key_env": KEY_ENV.get(vendor),
                     "key_present": key_present(vendor), "reachable": False,
                     "error": ""}
        if not row["key_present"]:
            row["error"] = f"{KEY_ENV.get(vendor)} not set"
        else:
            try:
                c = VendorClient(name, max_tokens=16)
                txt = c.complete("Reply with the single word OK.", "Ready?")
                row["reachable"] = bool(txt and txt.strip())
                row["sample"] = (txt or "").strip()[:40]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        out[f"{vendor}:{model}"] = row
        if verbose:
            mark = "ok  " if row["reachable"] else "FAIL"
            print(f"  [{mark}] {vendor}:{model}"
                  + (f"  -- {row['error']}" if row["error"] else ""))
    return out
