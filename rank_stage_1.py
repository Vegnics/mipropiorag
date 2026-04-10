import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pymupdf
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from question_extraction2 import extract_retrieval_keywords_semantic
from question_intention import get_intention_scores

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SLIDESPATH = "slides_merged.json"
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
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pooling(out, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embeds.append(emb)
    return torch.cat(all_embeds, dim=0)


def encode_one(text: str) -> torch.Tensor:
    return encode_texts([text])[0]


def cos_sim(q: torch.Tensor, x: torch.Tensor) -> float:
    return F.cosine_similarity(q.unsqueeze(0), x.unsqueeze(0), dim=-1).item()


QUESTION_KEYS = [
    "phrase_lookup",
    "entity_lookup",
    "numeric",
    "list",
    "definition",
    "property",
    "comparison",
    "structure",
    "purpose_why",
    "causal_reasoning",
    "process_steps",
    "mechanism_how",
    "visual_identification",
    "visual_alignment",
    "dataset_metric",
    "system_limitation",
    "compositional",
    "example_case"
]


def normalize_text(x):
    if x is None:
        return None
    x = str(x).strip()
    if not x:
        return None
    if x.lower() in {"none", "null", "n/a"}:
        return None
    return x


def precompute_qadesc_embeddings(
    slides_dict: dict,
    active_qtypes: List[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if active_qtypes is None:
        active_qtypes = QUESTION_KEYS

    emb_store = {}

    for slide_id, slide in slides_dict.items():
        emb_store[slide_id] = {}
        qa_descs = slide.get("qa_descs", {})

        valid_qtypes = []
        valid_texts = []

        for qkey in active_qtypes:
            text = normalize_text(qa_descs.get(qkey))
            if text is not None:
                valid_qtypes.append(qkey)
                valid_texts.append(text)

        if len(valid_texts) == 0:
            continue

        embeds = encode_texts(valid_texts)

        for qkey, emb in zip(valid_qtypes, embeds):
            emb_store[slide_id][qkey] = emb

    return emb_store


def rank_slides_by_weighted_qadesc(
    slides_dict: dict,
    qadesc_embeddings: Dict[str, Dict[str, torch.Tensor]],
    query_embedding: torch.Tensor,
    intention_scores: dict,
    active_qtypes: List[str] = None,
    normalize_by_used_weights: bool = True,
):
    if active_qtypes is None:
        active_qtypes = QUESTION_KEYS

    ranked = []

    for slide_id, slide in slides_dict.items():
        qa_descs = slide.get("qa_descs", {})
        total_score = 0.0
        used_weight_sum = 0.0
        matched_fields = []

        slide_embs = qadesc_embeddings.get(slide_id, {})

        for qkey in active_qtypes:
            weight = float(intention_scores.get(qkey, 0.0))
            if weight <= 0:
                continue

            desc_text = normalize_text(qa_descs.get(qkey))
            desc_emb = slide_embs.get(qkey)

            if desc_text is None or desc_emb is None:
                continue

            sim = cos_sim(query_embedding, desc_emb)
            weighted_sim = weight * sim
            #weighted_sim = weight * sim

            total_score += weighted_sim
            used_weight_sum += weight

            matched_fields.append({
                "qtype": qkey,
                "weight": weight,
                "sim": sim,
                "weighted_sim": weighted_sim,
                "text": desc_text
            })

        if normalize_by_used_weights and used_weight_sum > 0:
            final_score = total_score / used_weight_sum
        else:
            final_score = total_score

        ranked.append({
            "slide_id": int(slide_id),
            "score": final_score,
            "raw_score": total_score,
            "used_weight_sum": used_weight_sum,
            "matched_fields": sorted(
                matched_fields,
                key=lambda x: x["weighted_sim"],
                reverse=True
            )
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def load_questions(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    with open(SLIDESPATH, "r", encoding="utf-8") as f:
        slidesdict = json.load(f)

    # You can restrict to the more useful semantic fields if needed
    ACTIVE_QTYPES = [
        "phrase_lookup",
        "definition",
        "comparison",
        "purpose_why",
        "mechanism_how",
        "visual_alignment",
        "system_limitation",
        "example_case",
        "structure",
        "process_steps",
        "causal_reasoning",
        "compositional",
    ]

    print("Precomputing qa_desc embeddings...")
    qadesc_embeddings = precompute_qadesc_embeddings(
        slidesdict,
        active_qtypes=ACTIVE_QTYPES
    )

    #QUESTION = ""
    # Example:
    #QUESTION = "Why do we intentionally block the architecture from evaluating certain elements of the sequence during training and generation?"
    QUESTION = "What specific term is mathematically equated to the process of generating outputs without any guiding prompts or context?"
    kwords = ", ".join(extract_retrieval_keywords_semantic(QUESTION, encode_texts, top_k=8))
    intention_scores = get_intention_scores(QUESTION)

    print(f"Question: {QUESTION}")
    print(f"Extracted keywords: {kwords}")
    print(f"Intention scores: {intention_scores}")
    question_embedding = encode_one(QUESTION)
    kw_embedding = encode_one(kwords)
    #query_embedding = encode_one(kwords + " ." +QUESTION)
    query_embedding = 0.8*question_embedding #+ 0.2*kw_embedding
    query_embedding = F.normalize(query_embedding, p=2, dim=0)

    ranked = rank_slides_by_weighted_qadesc(
        slides_dict=slidesdict,
        qadesc_embeddings=qadesc_embeddings,
        query_embedding=query_embedding,
        intention_scores=intention_scores,
        active_qtypes=ACTIVE_QTYPES,
        normalize_by_used_weights=True,
    )

    print("\nTop ranked slides:")
    for r in ranked[:10]:
        print(f"Rank {ranked.index(r) + 1}: \nSlide {r['slide_id']} | score={r['score']:.4f} | raw={r['raw_score']:.4f}")
        for m in r["matched_fields"][:3]:
            print(
                f"  - {m['qtype']:<20} "
                f"sim={m['sim']:.4f} "
                f"w={m['weight']:.4f} "
                f"wsim={m['weighted_sim']:.4f}"
            )
            print(f"    text: {m['text']}")