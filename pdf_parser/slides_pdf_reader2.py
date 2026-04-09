from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re
from urllib.parse import urlparse
from lang_utils.raw_text_ops2 import clean_slide_text,clean_page_text,remove_isolated_characters
from visual.ocr import OCR_Reader2 as OCR_Reader
import pymupdf
from PIL import Image
import numpy as np
import json
import io
# =========================================================
# Schema helpers
# =========================================================

FIELD_TRUST = {
    "title": 1.00,
    "body": 1.00,
    "ocr": 0.75,
    "desc": 0.60,
    "reference_title": 0.95,
    "reference_venue": 0.85,
    "reference_year": 0.80,
    "reference_url": 0.75,
    "reference_raw": 0.85,
}

MIN_CHARS = {
    "title": 2,
    "body": 8,
    "ocr": 8,
    "desc": 12,
    "reference_title": 4,
    "reference_venue": 2,
    "reference_year": 4,
    "reference_url": 8,
    "reference_raw": 8,
}


def empty_slide_record(slide_id: int) -> Dict[str, Any]:
    return {
        "slide_id": slide_id,

        # full-text fields
        "title_text": "",
        "body_text": "",
        "ocr_text": "",
        "desc_text": "",
        "reference": None,

        # chunked fields
        "title_chunks": [],
        "body_chunks": [],
        "ocr_chunks": [],
        "desc_chunks": [],
        "reference_chunks": [],

        # unified view
        "chunks": [],
    }


def make_chunk(
    slide_id: int,
    field: str,
    text: str,
    order: int,
    sub_id: int,
    trust: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trust = FIELD_TRUST.get(field, 0.7) if trust is None else trust
    return {
        "chunk_id": f"{slide_id}_{field}_{sub_id}",
        "slide_id": slide_id,
        "field": field,
        "text": text,
        "order": order,
        "trust": trust,
        "meta": {} if meta is None else meta,
    }


# =========================================================
# Cleaning / splitting helpers
# =========================================================

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_chunk_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = normalize_spaces(text)
    return text


def looks_like_noise(text: str, min_chars: int = 8) -> bool:
    t = clean_chunk_text(text)
    if len(t) < min_chars:
        return True

    # too little alphabetic information
    alpha = sum(ch.isalpha() for ch in t)
    if alpha == 0:
        return True

    # isolated symbols or counters
    if re.fullmatch(r"[\W_]+", t):
        return True

    return False


def split_sentences_simple(text: str) -> List[str]:
    """
    Lightweight sentence splitter that avoids external deps.
    Good enough for slide/OCR/description chunks.
    """
    text = clean_chunk_text(text)
    if not text:
        return []

    # Protect common abbreviations a bit
    protected = text
    protected = protected.replace("e.g.", "eg")
    protected = protected.replace("i.e.", "ie")
    protected = protected.replace("etc.", "etc")

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|(?<=\.)\s+(?=•)", protected)
    parts = [clean_chunk_text(p) for p in parts if clean_chunk_text(p)]
    return parts


def split_into_two_sentence_chunks(text: str, min_chars: int = 8) -> List[str]:
    sents = split_sentences_simple(text)
    if not sents:
        return []

    out = []
    i = 0
    while i < len(sents):
        if i + 1 < len(sents):
            merged = clean_chunk_text(sents[i] + " " + sents[i + 1])
            out.append(merged)
            i += 2
        else:
            out.append(sents[i])
            i += 1

    out = [x for x in out if not looks_like_noise(x, min_chars=min_chars)]
    return out


def split_lines(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [clean_chunk_text(x) for x in text.split("\n")]
    return [x for x in lines if x]


def split_bullets_or_lines(text: str) -> List[str]:
    """
    Useful for body text that may contain bullets or one-item-per-line content.
    """
    if not text:
        return []

    text = text.replace("•", "\n• ")
    lines = split_lines(text)

    out: List[str] = []
    for line in lines:
        line = re.sub(r"^\s*[•\-]\s*", "", line)
        line = clean_chunk_text(line)
        if line:
            out.append(line)
    return out


def merge_short_neighbors(chunks: List[str], max_words: int = 40) -> List[str]:
    """
    Merge adjacent very short chunks into more meaningful units.
    """
    out: List[str] = []
    buf = ""

    for ch in chunks:
        candidate = ch if not buf else f"{buf} {ch}"
        if len(candidate.split()) <= max_words:
            buf = candidate
        else:
            if buf:
                out.append(clean_chunk_text(buf))
            buf = ch

    if buf:
        out.append(clean_chunk_text(buf))

    return out


# =========================================================
# Reference helpers
# =========================================================

def detect_reference_type(ref: Any) -> str:
    if ref is None:
        return "none"

    if isinstance(ref, dict):
        if ref.get("url"):
            return "url"
        if ref.get("title"):
            return "paper"
        return "structured"

    if isinstance(ref, str):
        if re.search(r"https?://", ref):
            return "url"
        return "raw"

    return "unknown"


def normalize_reference_obj(ref: Any) -> Optional[Dict[str, Any]]:
    """
    Accepts:
    - None
    - raw string
    - current parsed dict with title/conference/year
    - dict with url
    """
    if ref is None:
        return None

    if isinstance(ref, dict):
        out = {
            "type": detect_reference_type(ref),
            "raw": ref.get("raw"),
            "title": ref.get("title"),
            "venue": ref.get("conference") or ref.get("venue"),
            "year": ref.get("year"),
            "url": ref.get("url"),
        }
        if out["raw"] is None:
            raw_parts = [x for x in [out["title"], out["venue"], out["year"]] if x]
            out["raw"] = ", ".join(map(str, raw_parts)) if raw_parts else None
        return out

    if isinstance(ref, str):
        text = clean_chunk_text(ref)
        url_match = re.search(r"https?://\S+", text)
        return {
            "type": "url" if url_match else "raw",
            "raw": text,
            "title": None,
            "venue": None,
            "year": None,
            "url": url_match.group(0) if url_match else None,
        }

    return None


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


# =========================================================
# Chunk builders
# =========================================================

def build_title_chunks(slide_id: int, title_text: str) -> List[Dict[str, Any]]:
    title_text = clean_chunk_text(title_text)
    if looks_like_noise(title_text, min_chars=MIN_CHARS["title"]):
        return []
    return [make_chunk(slide_id, "title", title_text, order=0, sub_id=0)]


def build_body_chunks_from_blocks(
    slide_id: int,
    body_blocks: List[Dict[str, Any]],
    start_order: int = 100,
) -> List[Dict[str, Any]]:
    """
    body_blocks format:
    [
        {"text": "...", "origin": (x, y), "block_idx": 0},
        ...
    ]
    """
    chunks: List[Dict[str, Any]] = []
    sub_id = 0
    order = start_order

    for b in body_blocks:
        raw = remove_isolated_characters(clean_chunk_text(b.get("text", "")))
        if looks_like_noise(raw, min_chars=MIN_CHARS["body"]):
            continue

        # If block is short enough, keep it as one chunk
        if len(raw.split()) <= 40:
            chunks.append(
                make_chunk(
                    slide_id,
                    "body",
                    raw,
                    order=order,
                    sub_id=sub_id,
                    meta={
                        "origin": b.get("origin"),
                        "block_idx": b.get("block_idx"),
                        "source": "text_block",
                    },
                )
            )
            sub_id += 1
            order += 1
            continue

        # Otherwise split into bullet-like or sentence-like units
        candidates = split_bullets_or_lines(raw)
        if len(candidates) <= 1:
            candidates = split_into_two_sentence_chunks(raw, min_chars=MIN_CHARS["body"])
        else:
            candidates = merge_short_neighbors(candidates, max_words=40)

        for piece in candidates:
            piece = clean_chunk_text(piece)
            if looks_like_noise(piece, min_chars=MIN_CHARS["body"]):
                continue
            chunks.append(
                make_chunk(
                    slide_id,
                    "body",
                    piece,
                    order=order,
                    sub_id=sub_id,
                    meta={
                        "origin": b.get("origin"),
                        "block_idx": b.get("block_idx"),
                        "source": "text_block_split",
                    },
                )
            )
            sub_id += 1
            order += 1

    return chunks


def build_body_chunks_from_text(
    slide_id: int,
    body_text: str,
    start_order: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fallback if you only have merged body text and not original blocks.
    """
    body_text = remove_isolated_characters(clean_chunk_text(body_text))
    #body_text = clean_chunk_text(body_text)
    if looks_like_noise(body_text, min_chars=MIN_CHARS["body"]):
        return []

    candidates = split_bullets_or_lines(body_text)
    if len(candidates) <= 1:
        candidates = split_into_two_sentence_chunks(body_text, min_chars=MIN_CHARS["body"])
    else:
        candidates = merge_short_neighbors(candidates, max_words=40)

    out = []
    for i, piece in enumerate(candidates):
        if looks_like_noise(piece, min_chars=MIN_CHARS["body"]):
            continue
        out.append(
            make_chunk(
                slide_id,
                "body",
                piece,
                order=start_order + i,
                sub_id=i,
                meta={"source": "body_text_fallback"},
            )
        )
    return out

"""
def build_ocr_chunks(
    slide_id: int,
    ocr_text: str,
    start_order: int = 200,
) -> List[Dict[str, Any]]:
    ocr_text = clean_chunk_text(ocr_text)
    if looks_like_noise(ocr_text, min_chars=MIN_CHARS["ocr"]):
        return []

    candidates = split_into_two_sentence_chunks(ocr_text, min_chars=MIN_CHARS["ocr"])
    if not candidates:
        candidates = merge_short_neighbors(split_bullets_or_lines(ocr_text), max_words=30)

    out = []
    for i, piece in enumerate(candidates):
        if looks_like_noise(piece, min_chars=MIN_CHARS["ocr"]):
            continue
        out.append(
            make_chunk(
                slide_id,
                "ocr",
                piece,
                order=start_order + i,
                sub_id=i,
                meta={"source": "ocr_text"},
            )
        )
    return out
"""

def build_ocr_chunks_from_lines(
    slide_id: int,
    ocr_lines: list[str],
    start_order: int = 200,
    max_words: int = 28,
):
    chunks = []
    buf = []
    sub_id = 0
    order = start_order

    def line_is_garbage(line: str) -> bool:
        line = line.strip()
        if len(line) < 6:
            return True
        alpha = sum(ch.isalpha() for ch in line)
        if alpha / max(len(line), 1) < 0.35:
            return True
        if len(re.findall(r"[A-Za-z]{3,}", line)) == 0:
            return True
        return False

    clean_lines = [ln.strip() for ln in ocr_lines if ln.strip()]
    clean_lines = [ln for ln in clean_lines if not line_is_garbage(ln)]

    for line in clean_lines:
        candidate = " ".join(buf + [line]).strip()
        if len(candidate.split()) <= max_words:
            buf.append(line)
        else:
            if buf:
                chunks.append(
                    make_chunk(
                        slide_id,
                        "ocr",
                        " ".join(buf),
                        order=order,
                        sub_id=sub_id,
                        meta={"source": "ocr_lines"}
                    )
                )
                sub_id += 1
                order += 1
            buf = [line]

    if buf:
        chunks.append(
            make_chunk(
                slide_id,
                "ocr",
                " ".join(buf),
                order=order,
                sub_id=sub_id,
                meta={"source": "ocr_lines"}
            )
        )

    return chunks

def build_ocr_chunks(
    slide_id: int,
    ocr_text: str,
    start_order: int = 200,
) -> List[Dict[str, Any]]:
    ocr_text = ocr_text.strip()
    if not ocr_text:
        return []

    lines = [x.strip() for x in ocr_text.split("\n") if x.strip()]

    # drop tiny junk lines
    lines = [x for x in lines if len(x) >= 8]

    # merge neighboring short lines into chunks
    merged = []
    buf = ""
    for line in lines:
        candidate = line if not buf else f"{buf} {line}"
        if len(candidate.split()) <= 25:
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            buf = line
    if buf:
        merged.append(buf)

    out = []
    for i, piece in enumerate(merged):
        out.append(
            make_chunk(
                slide_id,
                "ocr",
                piece,
                order=start_order + i,
                sub_id=i,
                meta={"source": "ocr_lines"}
            )
        )
    return out

def build_desc_chunks(
    slide_id: int,
    desc_text: str,
    start_order: int = 300,
) -> List[Dict[str, Any]]:
    desc_text = clean_chunk_text(desc_text)
    if looks_like_noise(desc_text, min_chars=MIN_CHARS["desc"]):
        return []

    candidates = split_into_two_sentence_chunks(desc_text, min_chars=MIN_CHARS["desc"])
    if not candidates:
        candidates = [desc_text]

    out = []
    for i, piece in enumerate(candidates):
        if looks_like_noise(piece, min_chars=MIN_CHARS["desc"]):
            continue
        out.append(
            make_chunk(
                slide_id,
                "desc",
                piece,
                order=start_order + i,
                sub_id=i,
                meta={"source": "slide_description"},
            )
        )
    return out


def build_reference_chunks(
    slide_id: int,
    reference_obj: Optional[Dict[str, Any]],
    start_order: int = 400,
) -> List[Dict[str, Any]]:
    if not reference_obj:
        return []

    out: List[Dict[str, Any]] = []
    sub_id = 0
    order = start_order

    title = reference_obj.get("title")
    venue = reference_obj.get("venue")
    year = reference_obj.get("year")
    raw = reference_obj.get("raw")
    url = reference_obj.get("url")

    if title and not looks_like_noise(title, min_chars=MIN_CHARS["reference_title"]):
        out.append(make_chunk(slide_id, "reference_title", clean_chunk_text(title), order, sub_id))
        sub_id += 1
        order += 1

    if venue and not looks_like_noise(str(venue), min_chars=MIN_CHARS["reference_venue"]):
        out.append(make_chunk(slide_id, "reference_venue", clean_chunk_text(str(venue)), order, sub_id))
        sub_id += 1
        order += 1

    if year is not None:
        year_text = str(year)
        if not looks_like_noise(year_text, min_chars=MIN_CHARS["reference_year"]):
            out.append(make_chunk(slide_id, "reference_year", year_text, order, sub_id))
            sub_id += 1
            order += 1

    if url and not looks_like_noise(url, min_chars=MIN_CHARS["reference_url"]):
        out.append(
            make_chunk(
                slide_id,
                "reference_url",
                clean_chunk_text(url),
                order,
                sub_id,
                meta={"domain": domain_from_url(url)},
            )
        )
        sub_id += 1
        order += 1

    if raw and not looks_like_noise(raw, min_chars=MIN_CHARS["reference_raw"]):
        out.append(make_chunk(slide_id, "reference_raw", clean_chunk_text(raw), order, sub_id))

    return out


def rebuild_unified_chunks(slide_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []
    for key in ["title_chunks", "body_chunks", "ocr_chunks", "desc_chunks", "reference_chunks"]:
        chunks.extend(slide_record.get(key, []))
    chunks.sort(key=lambda x: x["order"])
    return chunks


# =========================================================
# Pipeline adapters for your current code
# =========================================================


def merged_text_from_block_chunks(block_chunks: List[Dict[str, Any]]) -> str:
    return "\n".join(x["text"] for x in block_chunks if x.get("text"))

def build_slide_record(
    slide_id: int,
    title_text: str,
    body_text: str,
    ocr_lines: List[str],
    desc_text: str = "",
    reference: Any = None,
    body_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rec = empty_slide_record(slide_id)

    rec["title_text"] = clean_chunk_text(title_text)
    ## Removing isolated chars  
    rec["body_text"] = remove_isolated_characters(clean_chunk_text(body_text))

    # keep OCR full text for debugging / storage
    clean_ocr_lines = [clean_chunk_text(x) for x in ocr_lines if x and clean_chunk_text(x)]
    rec["ocr_text"] = "\n".join(clean_ocr_lines)

    rec["desc_text"] = clean_chunk_text(desc_text)
    rec["reference"] = normalize_reference_obj(reference)

    rec["title_chunks"] = build_title_chunks(slide_id, rec["title_text"])

    if body_blocks:
        rec["body_chunks"] = build_body_chunks_from_blocks(slide_id, body_blocks)
    else:
        rec["body_chunks"] = build_body_chunks_from_text(slide_id, rec["body_text"])

    rec["ocr_chunks"] = build_ocr_chunks_from_lines(slide_id, clean_ocr_lines)
    rec["desc_chunks"] = build_desc_chunks(slide_id, rec["desc_text"])
    rec["reference_chunks"] = build_reference_chunks(slide_id, rec["reference"])

    rec["chunks"] = rebuild_unified_chunks(rec)
    return rec

#from __future__ import annotations

#from typing import Any, Dict, List, Optional, Tuple



# =========================================================
# Schema helpers
# =========================================================

FIELD_TRUST = {
    "title": 1.00,
    "body": 1.00,
    "ocr": 0.75,
    "desc": 0.60,
    "reference_title": 0.95,
    "reference_venue": 0.85,
    "reference_year": 0.80,
    "reference_url": 0.75,
    "reference_raw": 0.85,
}

MIN_CHARS = {
    "title": 2,
    "body": 8,
    "ocr": 8,
    "desc": 12,
    "reference_title": 4,
    "reference_venue": 2,
    "reference_year": 4,
    "reference_url": 8,
    "reference_raw": 8,
}


def empty_slide_record(slide_id: int) -> Dict[str, Any]:
    return {
        "slide_id": slide_id,

        # full-text fields
        "title_text": "",
        "body_text": "",
        "ocr_text": "",
        "desc_text": "",
        "reference": None,

        # chunked fields
        "title_chunks": [],
        "body_chunks": [],
        "ocr_chunks": [],
        "desc_chunks": [],
        "reference_chunks": [],

        # unified view
        "chunks": [],
    }


def make_chunk(
    slide_id: int,
    field: str,
    text: str,
    order: int,
    sub_id: int,
    trust: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trust = FIELD_TRUST.get(field, 0.7) if trust is None else trust
    return {
        "chunk_id": f"{slide_id}_{field}_{sub_id}",
        "slide_id": slide_id,
        "field": field,
        "text": text,
        "order": order,
        "trust": trust,
        "meta": {} if meta is None else meta,
    }


# =========================================================
# Cleaning / splitting helpers
# =========================================================

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_chunk_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = normalize_spaces(text)
    return text


def looks_like_noise(text: str, min_chars: int = 8) -> bool:
    t = clean_chunk_text(text)
    if len(t) < min_chars:
        return True

    # too little alphabetic information
    alpha = sum(ch.isalpha() for ch in t)
    if alpha == 0:
        return True

    # isolated symbols or counters
    if re.fullmatch(r"[\W_]+", t):
        return True

    return False


def split_sentences_simple(text: str) -> List[str]:
    """
    Lightweight sentence splitter that avoids external deps.
    Good enough for slide/OCR/description chunks.
    """
    text = clean_chunk_text(text)
    if not text:
        return []

    # Protect common abbreviations a bit
    protected = text
    protected = protected.replace("e.g.", "eg")
    protected = protected.replace("i.e.", "ie")
    protected = protected.replace("etc.", "etc")

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|(?<=\.)\s+(?=•)", protected)
    parts = [clean_chunk_text(p) for p in parts if clean_chunk_text(p)]
    return parts


def split_into_two_sentence_chunks(text: str, min_chars: int = 8) -> List[str]:
    sents = split_sentences_simple(text)
    if not sents:
        return []

    out = []
    i = 0
    while i < len(sents):
        if i + 1 < len(sents):
            merged = clean_chunk_text(sents[i] + " " + sents[i + 1])
            out.append(merged)
            i += 2
        else:
            out.append(sents[i])
            i += 1

    out = [x for x in out if not looks_like_noise(x, min_chars=min_chars)]
    return out


def split_lines(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [clean_chunk_text(x) for x in text.split("\n")]
    return [x for x in lines if x]


def split_bullets_or_lines(text: str) -> List[str]:
    """
    Useful for body text that may contain bullets or one-item-per-line content.
    """
    if not text:
        return []

    text = text.replace("•", "\n• ")
    lines = split_lines(text)

    out: List[str] = []
    for line in lines:
        line = re.sub(r"^\s*[•\-]\s*", "", line)
        line = clean_chunk_text(line)
        if line:
            out.append(line)
    return out


def merge_short_neighbors(chunks: List[str], max_words: int = 40) -> List[str]:
    """
    Merge adjacent very short chunks into more meaningful units.
    """
    out: List[str] = []
    buf = ""

    for ch in chunks:
        candidate = ch if not buf else f"{buf} {ch}"
        if len(candidate.split()) <= max_words:
            buf = candidate
        else:
            if buf:
                out.append(clean_chunk_text(buf))
            buf = ch

    if buf:
        out.append(clean_chunk_text(buf))

    return out


# =========================================================
# Reference helpers
# =========================================================

def detect_reference_type(ref: Any) -> str:
    if ref is None:
        return "none"

    if isinstance(ref, dict):
        if ref.get("url"):
            return "url"
        if ref.get("title"):
            return "paper"
        return "structured"

    if isinstance(ref, str):
        if re.search(r"https?://", ref):
            return "url"
        return "raw"

    return "unknown"


def normalize_reference_obj(ref: Any) -> Optional[Dict[str, Any]]:
    """
    Accepts:
    - None
    - raw string
    - current parsed dict with title/conference/year
    - dict with url
    """
    if ref is None:
        return None

    if isinstance(ref, dict):
        out = {
            "type": detect_reference_type(ref),
            "raw": ref.get("raw"),
            "title": ref.get("title"),
            "venue": ref.get("conference") or ref.get("venue"),
            "year": ref.get("year"),
            "url": ref.get("url"),
        }
        if out["raw"] is None:
            raw_parts = [x for x in [out["title"], out["venue"], out["year"]] if x]
            out["raw"] = ", ".join(map(str, raw_parts)) if raw_parts else None
        return out

    if isinstance(ref, str):
        text = clean_chunk_text(ref)
        url_match = re.search(r"https?://\S+", text)
        return {
            "type": "url" if url_match else "raw",
            "raw": text,
            "title": None,
            "venue": None,
            "year": None,
            "url": url_match.group(0) if url_match else None,
        }

    return None


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

# =========================================================
# Pipeline adapters for your current code
# =========================================================

def text_blocks_from_pymupdf_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preserve text-box chunks instead of collapsing them too early.
    This mirrors your current text_from_blocks logic but returns blocks.
    """
    out = []
    block_counter = 0

    for b in blocks:
        if b.get("type") != 0:
            continue

        full_text = ""
        origin = None

        for line in b.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            if origin is None and "origin" in spans[0]:
                origin = spans[0]["origin"]

            line_text = "".join(span.get("text", "") for span in spans)
            if line_text.strip():
                full_text += line_text + " "

        full_text = clean_chunk_text(full_text)
        if full_text:
            out.append({
                "text": full_text,
                "origin": origin,
                "block_idx": block_counter,
            })
            block_counter += 1

    out = sorted(out, key=lambda x: (
        x["origin"][1] if x["origin"] is not None else 10**9,
        x["origin"][0] if x["origin"] is not None else 10**9
    ))
    return out



######################################################
def split_reference_robust(ref: str):
    """
    Handles:
    - "..., NeurIPS, 2017"
    - "..., NIPS. 2017"
    """

    # split by comma OR period
    parts = re.split(r'[,.]', ref)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) < 3:
        return None

    # last element = year
    try:
        year = int(parts[-1])
    except:
        return None

    return {
        "title": ", ".join(parts[:-2]),
        "conference": parts[-2],
        "year": year
    }

def extract_source_block(text):
    pattern = r'Source:\s*(.*?\b\d{4})'
    matches = re.findall(pattern, text, re.IGNORECASE)
    results = []
    for ref in matches:
        ref = ref.strip().rstrip(" .")
        parsed = split_reference_robust(ref)
        if parsed:
            results.append(parsed)

    return results[0] if results else None


########################################################
# Main processing loop for your PDF



doc = pymupdf.open("AI.pdf")
ocr_reader = OCR_Reader()

slides_dict = {}

print(doc.page_count)
for k in range(5):
    page = doc.load_page(k)
    w = page.rect.x1
    h = page.rect.y1

    page_json = page.get_text(option="dict")
    blocks = page_json["blocks"]

    # -----------------------------------------------------
    # 1) Preserve text-box blocks first
    # -----------------------------------------------------
    body_blocks_raw = text_blocks_from_pymupdf_blocks(blocks)

    # merged text only for title/body extraction
    merged_text = merged_text_from_block_chunks(body_blocks_raw)

    # your existing title/body cleanup
    title, body_text = clean_slide_text(merged_text)

    # -----------------------------------------------------
    # 2) OCR from image blocks
    # -----------------------------------------------------
    imgblocks = [b for b in blocks if b["type"] == 1]
    all_ocr_lines = []
    imgtext = ""    

    for b in imgblocks:
        img = Image.open(io.BytesIO(b["image"])).convert("RGB")
        img = np.array(img)

        ocr_lines = ocr_reader.image2text_ocr(img)
        if ocr_lines:
            # keep line structure
            all_ocr_lines.extend([x.strip() for x in ocr_lines if x and x.strip()])

    #imgtext = clean_page_text(imgtext)

    # -----------------------------------------------------
    # 3) Optional slide description
    # -----------------------------------------------------
    # Keep empty for now if you are not generating descriptions yet
    desc_text = ""

    # -----------------------------------------------------
    # 4) Reference extraction
    # -----------------------------------------------------
    reference = extract_source_block(body_text)

    # If later you also detect URLs, you can merge them here
    # Example:
    # if reference is None:
    #     url_match = re.search(r'https?://\S+', body_text)
    #     if url_match:
    #         reference = {"url": url_match.group(0), "raw": url_match.group(0)}

    # -----------------------------------------------------
    # 5) Build structured slide record
    # -----------------------------------------------------
    slide_record = build_slide_record(
        slide_id=k,
        title_text=title,
        body_text=body_text,
        ocr_lines=all_ocr_lines,
        desc_text=desc_text,
        reference=reference,
        body_blocks=body_blocks_raw,
    )

    # -----------------------------------------------------
    # 6) Backward-compatible aliases (optional)
    # -----------------------------------------------------
    # Keeps old code working if it still expects these keys
    slide_record["title"] = slide_record["title_text"]
    slide_record["body"] = slide_record["body_text"]
    slide_record["imgtext"] = slide_record["ocr_text"]

    slides_dict[k] = slide_record

    # -----------------------------------------------------
    # 7) Debug print
    # -----------------------------------------------------
    print(
        f"Page {k+1}:\n"
        f"\tTitle: {slides_dict[k]['title_text']}\n"
        f"\tBody: {slides_dict[k]['body_text']}\n"
        f"\tImage Content: {slides_dict[k]['ocr_text']}\n"
        f"\t#Title chunks: {len(slides_dict[k]['title_chunks'])}\n"
        f"\t#Body chunks: {len(slides_dict[k]['body_chunks'])}\n"
        f"\t#OCR chunks: {len(slides_dict[k]['ocr_chunks'])}\n"
        f"\t#Desc chunks: {len(slides_dict[k]['desc_chunks'])}\n"
        f"\t#Reference chunks: {len(slides_dict[k]['reference_chunks'])}\n"
    )

    # -----------------------------------------------------
    # 8) Optional reference logic from your old code
    # -----------------------------------------------------
    if reference is not None:
        """
        slides_dict[k]["reference"] = normalize_reference_obj(reference)

        src_emb = encode_texts([
            slides_dict[k]["title_text"] + " " + slides_dict[k]["body_text"]
        ])[0]

        if (
            False
            and slides_dict[k]["reference"] is not None
            and slides_dict[k]["reference"].get("title") == "Attention is All You Need"
        ):
            with open("embeds_attention.json", "r") as f:
                dict_pdf = json.load(f)
            retrieved = dict_embedding_retrieve(dict_pdf, src_emb)
            print(f"Retrieved: {retrieved}")
            input("Press Enter ....")
        """
        pass

with open("slides_dict_new.json",'w') as f:
    json.dump(slides_dict,f)