import json
import re
from collections import defaultdict
from tkinter.messagebox import QUESTION
from typing import List, Dict, Any, Tuple

import pymupdf
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SLIDESPATH = "full_slides_dict.json"
QUESTIONPATH = "HW1_questions.json"


# -----------------------------
# Embedding model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def encode_texts(texts: List[str], batch_size: int = 32) -> torch.Tensor:
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pooling(out, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embeds.append(emb)
    return torch.cat(all_embeds, dim=0)


def cosine_sim(q: torch.Tensor, x: torch.Tensor) -> float:
    return F.cosine_similarity(q.unsqueeze(0), x.unsqueeze(0), dim=-1).item()


# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_local_units(text: str) -> List[str]:
    """
    Split body into local evidence units.
    """
    text = clean_text(text)
    if not text:
        return []

    # split bullets / numbered items / sentences
    parts = re.split(r"(?:•|\b\d+\.\s+|(?<=[.!?])\s+)", text)
    parts = [p.strip() for p in parts if p and p.strip()]

    # keep only reasonably informative chunks
    parts = [p for p in parts if len(p.split()) >= 3]

    # fallback
    if not parts:
        parts = [text]

    return parts


def tokenize_for_overlap(text: str) -> List[str]:
    text = text.lower()
    toks = re.findall(r"[a-z0-9\-]+", text)
    return toks


def extract_question_keywords(question: str) -> List[str]:
    toks = tokenize_for_overlap(question)
    stop = {
        "the", "a", "an", "of", "to", "and", "in", "is", "are", "was", "were",
        "that", "this", "what", "which", "when", "why", "how", "does", "do",
        "for", "with", "on", "by", "from", "it", "as", "be", "into", "at",
        "or", "if", "their", "they", "them", "can", "used", "use"
    }
    return [t for t in toks if t not in stop and len(t) > 2]


def lexical_overlap_score(question: str, text: str) -> float:
    q = set(extract_question_keywords(question))
    t = set(tokenize_for_overlap(text))
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def number_boost(question: str, text: str) -> float:
    #q_has_number_prompt = any(x in question.lower() for x in ["how many", "exactly how many", "what exact", "which specific"])
    q_has_number_prompt = any(x in question.lower() for x in ["how many", "exactly how many","how much","amount","number","specific number"])
    if not q_has_number_prompt:
        return 0.0
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    return 0.05 if nums else 0.0


def phrase_boost(question: str, text: str) -> float:
    q = question.lower()
    t = text.lower()
    boost = 0.0

    special_phrases = [
        "trivially true",
        "external knowledge",
        "openai",
        "red box",
        "question-answer pairs",
        "amethyst shard",
        "mastermind",
        "autoregressive generation",
        "masked self-attention",
        "positional encoding",
    ]
    for p in special_phrases:
        if p in q or p in t:
            if p in t:
                boost += 0.05

    return boost


def question_type_weights(question: str) -> Dict[str, float]:
    q = question.lower()

    weights = {
        "title": 0.9,
        "body": 1.0,
        "imgtext": 0.7,
        "desc": 0.9,
        "chunk": 1.2,
        "lexical": 0.5,
    }

    if any(x in q for x in ["what specific term", "what specific phrase"]):
        weights["chunk"] = 1.4
        weights["body"] = 1.1
        weights["desc"] = 0.45

    if any(x in q for x in ["how many", "exactly how many"]):
        weights["chunk"] = 1.25
        weights["lexical"] = 0.20

    if any(x in q for x in ["which conference", "which company", "which paper", "what exact name"]):
        weights["title"] = 1.2
        weights["chunk"] = 1.1

    if any(x in q for x in ["visual indicator", "visually", "diagram", "illustrates", "shown"]):
        weights["imgtext"] = 1.0
        weights["desc"] = 0.8

    if any(x in q for x in ["share", "different", "same", "versus", "fundamentally different"]):
        weights["body"] = 1.2
        weights["chunk"] = 1.35
        weights["desc"] = 0.45

    return weights


# -----------------------------
# Build slide index
# -----------------------------
def build_slide_index(slidesdict: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = []
    title_texts, body_texts, img_texts, desc_texts = [], [], [], []
    all_chunks = []
    chunk_owner = []

    for k, v in slidesdict.items():
        page = int(k) + 1
        title = clean_text(v.get("title", ""))
        body = clean_text(v.get("body", ""))
        imgtext = clean_text(v.get("imgtext", ""))
        desc = clean_text(v.get("slide_desc", ""))


        chunks = split_local_units(body)
        if imgtext:
            chunks.extend(split_local_units(imgtext))

        pages.append({
            "page": page,
            "title": title,
            "body": body,
            "imgtext": imgtext,
            "desc": desc,
            "chunks": chunks,
        })

        title_texts.append(title if title else "[EMPTY]")
        body_texts.append(body if body else "[EMPTY]")
        img_texts.append(imgtext if imgtext else "[EMPTY]")
        desc_texts.append(desc if desc else "[EMPTY]")

        for c in chunks:
            all_chunks.append(c)
            chunk_owner.append(page)

    title_embs = encode_texts(title_texts)
    body_embs = encode_texts(body_texts)
    img_embs = encode_texts(img_texts)
    desc_embs = encode_texts(desc_texts)
    chunk_embs = encode_texts(all_chunks) if all_chunks else torch.empty(0, 768)

    # attach embeddings
    for i, p in enumerate(pages):
        p["title_emb"] = title_embs[i]
        p["body_emb"] = body_embs[i]
        p["img_emb"] = img_embs[i]
        p["desc_emb"] = desc_embs[i]

    page_to_chunk_idxs = defaultdict(list)
    for idx, page in enumerate(chunk_owner):
        page_to_chunk_idxs[page].append(idx)

    return pages, chunk_embs, all_chunks, page_to_chunk_idxs


# -----------------------------
# Retrieval
# -----------------------------
def retrieve(question: str,
             pages: List[Dict[str, Any]],
             chunk_embs: torch.Tensor,
             all_chunks: List[str],
             page_to_chunk_idxs: Dict[int, List[int]],
             top_k_candidates: int = 30) -> List[Dict[str, Any]]:

    q_emb = encode_texts([question])[0]
    w = question_type_weights(question)

    # Stage A: coarse candidate generation
    coarse = []
    for p in pages:
        sim_title = cosine_sim(q_emb, p["title_emb"])
        sim_body = cosine_sim(q_emb, p["body_emb"])
        sim_img = cosine_sim(q_emb, p["img_emb"])
        sim_desc = cosine_sim(q_emb, p["desc_emb"])

        coarse_score = sum([
            w["title"] * sim_title,
            w["body"] * sim_body,
            w["imgtext"] * sim_img,
            w["desc"] * sim_desc,]
        )

        coarse.append({
            "page": p["page"],
            "coarse_score": coarse_score,
            "sim_title": sim_title,
            "sim_body": sim_body,
            "sim_img": sim_img,
            "sim_desc": sim_desc,
            "title": p["title"],
            "body": p["body"],
            "imgtext": p["imgtext"],
            "desc": p["desc"],
        })

    coarse.sort(key=lambda x: x["coarse_score"], reverse=True)
    candidates = coarse[:top_k_candidates]

    # Stage B: local evidence reranking
    reranked = []
    for c in candidates:
        page = c["page"]
        best_chunk = ""
        best_chunk_sim = -1.0

        for idx in page_to_chunk_idxs.get(page, []):
            s = cosine_sim(q_emb, chunk_embs[idx])
            if s > best_chunk_sim:
                best_chunk_sim = s
                best_chunk = all_chunks[idx]

        local_text = " ".join([c["title"], c["body"], c["imgtext"], c["desc"]]).strip()
        lex = lexical_overlap_score(question, local_text)
        nb = number_boost(question, local_text)
        pb = phrase_boost(question, local_text)

        final_score = sum([
            w["chunk"] * best_chunk_sim,
            w["body"] * c["sim_body"],
            w["title"] * c["sim_title"],
            w["imgtext"] * c["sim_img"],
            w["desc"] * c["sim_desc"],
        ]) + w["lexical"] * lex + nb + pb

        reranked.append({
            **c,
            "best_chunk": best_chunk,
            "best_chunk_sim": best_chunk_sim,
            "lexical_overlap": lex,
            "final_score": final_score,
        })

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

  data = [["Question","Slide number","Confidence","Content"]]
  with open('raw_answers.csv', mode='w', newline='') as file:
    slides = pymupdf.open('AI.pdf') ## Open slides PDF with pymupdf
    answerspdf = pymupdf.open() ## PDF containing question / answer page
    
    with open(SLIDESPATH, "r", encoding="utf-8") as f:
        slidesdict = json.load(f)

    with open(QUESTIONPATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    pages, chunk_embs, all_chunks, page_to_chunk_idxs = build_slide_index(slidesdict)

    #for i, q in enumerate(questions[:5], start=1):
    for i, q in enumerate(questions, start=1):
        question = q["question"]
        ranked = retrieve(question, pages, chunk_embs, all_chunks, page_to_chunk_idxs, top_k_candidates=30)

        print("=" * 100)
        print(f"Q{i}: {question}")
        for r, item in enumerate(ranked[:5], start=1):
            print(f"{r}. page={item['page']} final={item['final_score']:.4f} "
                  f"chunk={item['best_chunk_sim']:.4f} body={item['sim_body']:.4f} "
                  f"title={item['sim_title']:.4f} img={item['sim_img']:.4f} desc={item['sim_desc']:.4f}")
            print("TITLE:", item["title"])
            print("BEST_CHUNK:", item["best_chunk"][:250])
            print("-" * 100)
        
        respage = ranked[0]["page"]-1
        answerspdf.insert_pdf(slides, from_page=respage, to_page=respage)
		# get the newly inserted page
        new_page = answerspdf[-1]
		# add annotation
        rect = pymupdf.Rect(600, 50, 900, 250)
        annot = new_page.add_freetext_annot(
            rect,
            f"QUESTION: {question}",
            fontsize=18,
            text_color=(1, 0.3, 0),
            fill_color=(1, 1, 0.8),
        )
        annot.update()
        data.append([question,ranked[0]["page"],ranked[0]["final_score"],ranked[0]["best_chunk"]])
    writer = csv.writer(file)
    writer.writerows(data)
answerspdf.save("answers_annotated.pdf")
