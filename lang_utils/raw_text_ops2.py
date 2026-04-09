from pdf2image import convert_from_path
import os
import tqdm
import json
import re
import pymupdf
import unicodedata


import re
import unicodedata

def normalize_unicode(text: str) -> str:
    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "∗": "*",
        "•": " • ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def normalize_to_ascii(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_isolated_characters(text: str) -> str:
    """
    Much safer version:
    only remove isolated lowercase junk letters,
    but keep meaningful uppercase symbols like P, Q, A, B, I.
    """
    text = re.sub(r'\b[b-hj-z]\b', ' ', text)   # optional, conservative
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


COMMON_HEADERS = [
    "This Lecture Agenda",
    "Artificial Intelligence",
]

def fix_urls(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(https?://)\s+", r"\1", text)
    text = re.sub(r"\s+([/.?=&_-])\s*", r"\1", text)
    return text

def expand_bullet_artifacts(text: str) -> str:
    # common OCR/PDF artifacts in these slides:
    # nPreliminaries -> • Preliminaries
    # lActing -> - Acting
    text = re.sub(r"\bn(?=[A-Z])", " • ", text)
    text = re.sub(r"\bl(?=[A-Z])", " • ", text)
    return text

def separate_merged_sentences(text: str) -> str:
    # add spaces between lowercase/number and uppercase transitions
    text = re.sub(r"(?<=[a-z0-9\)])(?=[A-Z])", " ", text)
    # add spaces after punctuation if missing
    text = re.sub(r"(?<=[.:;,])(?=[A-Za-z])", " ", text)
    return text

def separate_merged_title(text: str) -> str:
    # add spaces between lowercase/number and uppercase transitions
    text = re.sub(r"(?<=[a-z0-9\)])(?=[A-Z])", " ", text)
    return text

def fix_common_ocr_noise(text: str) -> str:
    # remove extra spaces around punctuation
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)

    # normalize weird repeated punctuation
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\-{2,}", "-", text)

    # common slide artifacts
    text = text.replace(" Lets", "Let's")
    text = text.replace(" doesnt", " does not")
    text = text.replace(" doesn't", " does not")
    text = text.replace(" didn", " did not")
    text = text.replace(" didn't", " did not")
    text = text.replace(" doesn", " does not")
    text = text.replace(" cant", " cannot")
    text = text.replace(" can't", " cannot")
    text = text.replace(" shouldnt", " should not")
    text = text.replace(" shouldn", " should not")
    text = text.replace(" haven't", " have not")
    text = text.replace(" havent", " have not")
    # clean malformed slide counters like (/) or ()
    text = re.sub(r"\(\s*/\s*\)", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    return text

def remove_source_noise(text: str) -> str:
    # keep source info, but standardize it
    text = re.sub(r"\bSource:\s*", " Source: ", text)
    text = re.sub(r"\s+Source:\s+", " Source: ", text)
    return text

def clean_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

"""
def split_into_lines(text: str) -> str:
    # make bullets and source easier to retrieve
    print(re.sub(r"\s*•\s*", "SUBBB: ", text))
    text = re.sub(r"\s*•\s*", "\n• ", text)
    text = re.sub(r"\s*-\s*", "\n• ", text)
    text = re.sub(r"\s+Source:\s+", "\nSource: ", text)
    return text.strip()
"""

def split_into_lines(text: str) -> str:
    # normalize CRLF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # split only real line-start bullets
    text = re.sub(r'^\s*-\s+', '• ', text, flags=re.MULTILINE)
    # source on its own line
    text = re.sub(r'\s+Source:\s+', '\nSource: ', text)

    return text.strip()

def drop_low_value_pages(text: str) -> bool:
    # discard empty / nearly empty slides
    stripped = re.sub(r"\s+", "", text)
    return len(stripped) < 8

def dedupe_repeated_phrases(text: str) -> str:
    # remove immediate duplicated phrases
    words = text.split()
    if not words:
        return text

    out = []
    i = 0
    while i < len(words):
        if i + 3 < len(words) and words[i:i+2] == words[i+2:i+4]:
            out.extend(words[i:i+2])
            i += 4
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)



def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fix_broken_words(text):
    # join single-letter splits like "T est" → "Test"
    text = re.sub(r'\b([A-Za-z])\s+([a-z])', r'\1\2', text)
    # join common split words (more aggressive)
    text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)
    return text


def remove_noise(text):
    # remove isolated symbols
    text = re.sub(r'[^\w\s.,:;()\-/%]', '', text)
    # remove repeated punctuation
    text = re.sub(r'\.{2,}', '.', text)
    # remove isolated characters
    text = re.sub(r'\s[A-Za-z]\s', '', text)
    return text

def remove_slide_artifacts(text):
    # remove page numbers (lines that are just numbers)
    text = re.sub(r'\b\d+\b', '', text)
    
    # remove common slide headers
    text = re.sub(r'AI Weekly', '', text)
    
    return text


def remove_chinese(text):
    has_chinese = re.search(r'[\u4e00-\u9fff]', text) is not None
    cleaned_text = re.sub(r'[\u4e00-\u9fff]', '', text)
    #if has_chinese:
    #    cleaned_text += "\nOne Chinese inscription is written."
    return cleaned_text


def normalize_case(text):
    return text.lower()


def remove_page_numbers(text):
    # remove lines that contain only numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    return text

def merge_broken_titles(text):
    lines = text.split('\n')
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # if line starts with "-" → continuation
        if line.startswith('-'):
            buffer += " " + line.lstrip('-').strip()
        else:
            if buffer:
                merged.append(buffer)
            buffer = line

    if buffer:
        merged.append(buffer)

    return '\n'.join(merged)

def fix_broken_words(text):
    # merge uppercase splits like "Neur IPS"
    text = re.sub(r'([A-Za-z]+)\s+([A-Z][a-z]+)', r'\1\2', text)

    # fix common ML terms manually (safe)
    replacements = {
        "Neur IPS": "NeurIPS",
        "ICLR ": "ICLR ",
        "CVPR ": "CVPR ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def remove_trailing_numbers(text):
    # remove numbers at end of line (slide numbers)
    text = re.sub(r'\s+\d+$', '', text, flags=re.MULTILINE)
    return text

def clean_titles(text):
    _text = text.replace("-\n","-")
    parts = _text.split("\n", maxsplit=1)
    title = parts[0].strip()
    title = normalize_whitespace(title)
    body = parts[1].strip() if len(parts) > 1 else ""
    return title, body

def clean_slide_text(text: str):
    text = remove_chinese(text)
    title, text = clean_titles(text)

    title = normalize_unicode(title)
    title = normalize_to_ascii(title)
    title = fix_common_ocr_noise(title)
    title = normalize_whitespace(title)

    text = remove_page_numbers(text)
    text = normalize_unicode(text)       # moved before ASCII
    text = normalize_to_ascii(text)      # now safer
    text = fix_urls(text)
    text = fix_common_ocr_noise(text)
    text = remove_source_noise(text)
    text = dedupe_repeated_phrases(text)
    text = normalize_whitespace(text)

    # optional: only if really needed
    # text = remove_isolated_characters(text)

    return title, text


def clean_page_text(text: str):
    text = remove_chinese(text)
    text = remove_page_numbers(text)
    text = normalize_unicode(text)       # moved before ASCII
    text = normalize_to_ascii(text)
    text = fix_urls(text)
    text = fix_common_ocr_noise(text)
    text = remove_source_noise(text)
    text = dedupe_repeated_phrases(text)
    text = normalize_whitespace(text)

    # optional: only if really needed
    # text = remove_isolated_characters(text)

    return text
