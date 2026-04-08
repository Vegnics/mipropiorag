import os
import re
import requests
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from urllib.parse import quote_plus

ARXIV_API = "http://export.arxiv.org/api/query"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def search_arxiv_by_title(title: str, max_results: int = 5):
    """
    Search arXiv using title field only: ti:"..."
    Returns best matching entry dict or None.
    """
    query = f'ti:"{title}"'
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }

    r = requests.get(ARXIV_API, params=params, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    entries = root.findall("atom:entry", ATOM_NS)

    if not entries:
        return None

    candidates = []
    for entry in entries:
        entry_title = entry.findtext("atom:title", default="", namespaces=ATOM_NS).strip()
        entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()

        pdf_url = None
        for link in entry.findall("atom:link", ATOM_NS):
            title_attr = link.attrib.get("title", "")
            href = link.attrib.get("href", "")
            if title_attr == "pdf":
                pdf_url = href
                break

        score = similarity(title, entry_title)
        candidates.append({
            "score": score,
            "title": entry_title,
            "id": entry_id,
            "pdf_url": pdf_url,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]

def download_pdf(url: str, output_path: str):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def find_and_download_arxiv(title: str, out_dir: str = "papers", min_similarity: float = 0.75):
    os.makedirs(out_dir, exist_ok=True)

    result = search_arxiv_by_title(title)
    if result is None:
        print("No arXiv result found.")
        return None

    print("Best arXiv match:")
    print("  title:", result["title"])
    print("  score:", round(result["score"], 3))
    print("  abs  :", result["id"])
    print("  pdf  :", result["pdf_url"])

    if result["score"] < min_similarity:
        print("Match score too low. Skipping download.")
        return result

    if not result["pdf_url"]:
        print("No PDF link found in arXiv entry.")
        return result

    filename = safe_filename(result["title"]) + ".pdf"
    output_path = os.path.join(out_dir, filename)
    download_pdf(result["pdf_url"], output_path)

    print("Downloaded to:", output_path)
    result["path"] = output_path
    return result

def create_dict_from_pdf(pdf_path):
    with open(SLIDESPATH, "r", encoding="utf-8") as file:
        slidesdict = json.load(file)



# Example
if __name__ == "__main__":
    find_and_download_arxiv("Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent")