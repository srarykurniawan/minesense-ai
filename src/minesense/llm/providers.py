"""LLM provider pluggable.

- extractive : tanpa LLM/jaringan. Menyusun jawaban dari kalimat paling relevan di konteks (+ sitasi).
               Berguna untuk dev, CI, dan fallback saat LLM down (graceful degradation).
- anthropic  : Claude via Messages API.
- openai     : endpoint OpenAI-compatible (OpenAI, vLLM, LM Studio, TGI, model fine-tune Anda).
- ollama     : Ollama lokal (memakai endpoint /v1 OpenAI-compatible). Cocok untuk model GGUF hasil fine-tune.
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod

import httpx

from minesense.config import Settings
from minesense.logging_utils import get_logger
from minesense.rag.text import split_sentences, tokenize

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


class LLM(ABC):
    name = "base"

    @abstractmethod
    def generate(self, system: str, user: str, *, max_tokens: int | None = None,
                 temperature: float | None = None, json_mode: bool = False) -> str: ...


def _retry(fn, attempts: int = 3, base: float = 1.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last = e
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (408, 429, 500, 502, 503, 504, 529):
                raise LLMError(f"HTTP {e.response.status_code}: {e.response.text[:300]}") from e
            last = e
        time.sleep(base * (2 ** i))
    raise LLMError(f"LLM gagal setelah {attempts} percobaan: {last}")


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, s: Settings):
        if not s.llm_api_key:
            raise LLMError("MINESENSE_LLM_API_KEY wajib diisi untuk provider anthropic")
        self.s = s
        self.client = httpx.Client(base_url=s.llm_base_url or "https://api.anthropic.com", timeout=s.llm_timeout_s,
                                   headers={"x-api-key": s.llm_api_key, "anthropic-version": "2023-06-01"})

    def generate(self, system, user, *, max_tokens=None, temperature=None, json_mode=False):
        body = {"model": self.s.llm_model, "max_tokens": max_tokens or self.s.llm_max_tokens,
                "temperature": self.s.llm_temperature if temperature is None else temperature,
                "system": system, "messages": [{"role": "user", "content": user}]}

        def call():
            r = self.client.post("/v1/messages", json=body)
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
        return _retry(call)


class OpenAICompatLLM(LLM):
    name = "openai"

    def __init__(self, s: Settings, default_base: str = "https://api.openai.com/v1"):
        self.s = s
        headers = {"Authorization": f"Bearer {s.llm_api_key}"} if s.llm_api_key else {}
        self.client = httpx.Client(base_url=s.llm_base_url or default_base, timeout=s.llm_timeout_s, headers=headers)

    def generate(self, system, user, *, max_tokens=None, temperature=None, json_mode=False):
        body = {"model": self.s.llm_model, "max_tokens": max_tokens or self.s.llm_max_tokens,
                "temperature": self.s.llm_temperature if temperature is None else temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        def call():
            r = self.client.post("/chat/completions", json=body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        return _retry(call)


class OllamaLLM(OpenAICompatLLM):
    name = "ollama"

    def __init__(self, s: Settings):
        super().__init__(s, default_base="http://ollama:11434/v1")


class ExtractiveLLM(LLM):
    """Fallback deterministik: memilih kalimat konteks dengan overlap token tertinggi terhadap pertanyaan."""

    name = "extractive"

    def generate(self, system, user, *, max_tokens=None, temperature=None, json_mode=False):
        if json_mode:
            return "{}"
        q = re.search(r"PERTANYAAN:\s*(.*)", user, re.S)
        question = q.group(1).strip() if q else user
        q_toks = set(tokenize(question))
        blocks = re.findall(r"\[(\d+)\]([^\n]*)\n(.*?)(?=\n\[\d+\]|\nPERTANYAAN:|\Z)", user, re.S)
        scored = []
        for n, header, body in blocks:
            h_toks = set(tokenize(header))
            for rank, sent in enumerate(split_sentences(body)):
                toks = set(tokenize(sent))
                if not toks:
                    continue
                overlap = len(q_toks & toks) + 0.5 * len(q_toks & (h_toks - toks))
                if len(q_toks & toks):
                    # normalisasi panjang (kalimat padat lebih disukai), bonus angka, penalti peringkat chunk
                    score = (overlap / (len(toks) ** 0.5) + 0.15 * bool(re.search(r"\d", sent))
                             - 0.03 * int(n) - 0.005 * rank)
                    scored.append((score, int(n), sent))
        if not scored:
            return "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang tersedia."
        scored.sort(key=lambda x: -x[0])
        picked, seen = [], set()
        for _, n, sent in scored:
            if sent in seen:
                continue
            seen.add(sent)
            picked.append(f"- {sent.rstrip('.')}. [{n}]")
            if len(picked) >= 4:
                break
        return "Berdasarkan dokumen terkait:\n" + "\n".join(picked)


def build_llm(s: Settings) -> LLM:
    match s.llm_provider:
        case "anthropic":
            return AnthropicLLM(s)
        case "openai":
            return OpenAICompatLLM(s)
        case "ollama":
            return OllamaLLM(s)
        case _:
            return ExtractiveLLM()


def parse_json(text: str) -> dict:
    """Ambil objek JSON pertama dari output LLM (toleran terhadap ```json fences```)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("tidak ada JSON")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON tidak lengkap")
