"""Loader dokumen + chunking sadar-struktur (heading markdown) dengan sliding window."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @property
    def embed_text(self) -> str:
        """Teks yang di-embed: judul + section memberi konteks pada chunk pendek."""
        return f"{self.title} | {self.section}\n{self.text}"


_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install extra 'pdf': pip install minesense[pdf]") from e
        reader = PdfReader(str(path))
        text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        meta: dict = {}
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        meta = {}
        m = _FM.match(text)
        if m:
            meta = yaml.safe_load(m.group(1)) or {}
            text = text[m.end():]
    doc_id = str(meta.get("doc_id") or path.stem)
    title = str(meta.get("title") or path.stem.replace("_", " "))
    meta = {k: str(v) for k, v in meta.items()}
    meta.setdefault("source", path.name)
    return Document(doc_id=doc_id, title=title, text=text.strip(), metadata=meta)


def load_corpus(directory: Path) -> list[Document]:
    exts = {".md", ".txt", ".pdf"}
    return [load_document(p) for p in sorted(directory.rglob("*")) if p.suffix.lower() in exts]


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Pisah berdasarkan heading markdown (#, ##, ###). Return [(section_path, body)]."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(stack) or "Umum", body))
        buf.clear()

    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = stack[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    # potong pada batas paragraf/baris bila memungkinkan
    parts, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
            elif (sp_end := text.rfind(" ", start, end)) > start + size // 2:
                end = sp_end
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        sp = text.find(" ", nxt)  # mulai overlap di awal kata, bukan di tengah kata
        start = sp + 1 if 0 <= sp < end else nxt
    return [p for p in parts if p]


def chunk_document(doc: Document, size: int = 700, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in _split_sections(doc.text):
        # gabung section terlalu pendek? Biarkan — section pendek = fakta padat, bagus untuk presisi.
        for piece in _window(body, size, overlap):
            idx = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{idx:03d}", doc_id=doc.doc_id, title=doc.title,
                section=section.split(" > ")[-1], text=piece, metadata=dict(doc.metadata),
            ))
    return chunks
