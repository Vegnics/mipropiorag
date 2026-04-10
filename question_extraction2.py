import spacy
import re
import torch
import torch.nn.functional as F
from typing import List, Tuple

from transformers import AutoModel, AutoTokenizer

nlp = spacy.load("en_core_web_sm")

QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose"
}

LIGHT_VERBS = {"be", "do", "have", "use", "show", "make", "take", "give", "put"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def clean_text(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def dedup_preserve_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        x = clean_text(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def content_tokens(span) -> List[str]:
    toks = []
    for t in span:
        lemma = t.lemma_.lower()
        if t.is_stop or t.is_punct or t.like_num:
            continue
        if lemma in QUESTION_WORDS:
            continue
        if len(lemma) <= 2:
            continue
        toks.append(lemma)
    return toks


def extract_candidate_phrases(text: str) -> List[str]:
    doc = nlp(text)
    candidates = []

    # 1) noun chunks
    for chunk in doc.noun_chunks:
        toks = content_tokens(chunk)
        if toks:
            candidates.append(" ".join(toks))

    # 2) important nouns / proper nouns
    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"}:
            toks = content_tokens([tok])
            if toks:
                candidates.append(toks[0])

    # 3) selective verbs
    for tok in doc:
        if tok.pos_ == "VERB":
            lemma = tok.lemma_.lower()
            if lemma not in LIGHT_VERBS and not tok.is_stop:
                candidates.append(lemma)

    # 4) verb + object
    for tok in doc:
        if tok.pos_ != "VERB":
            continue
        v = tok.lemma_.lower()
        if v in LIGHT_VERBS or tok.is_stop:
            continue

        for child in tok.children:
            if child.dep_ in {"dobj", "obj", "pobj", "attr"}:
                obj = content_tokens([child])
                if obj:
                    candidates.append(f"{v} {obj[0]}")

    # 5) adjacent content bigrams from original question
    content = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct and not t.like_num]
    for i in range(len(content) - 1):
        if len(content[i]) > 2 and len(content[i + 1]) > 2:
            candidates.append(f"{content[i]} {content[i+1]}")

    candidates = dedup_preserve_order(candidates)
    return candidates


def mmr_select(
    question_emb: torch.Tensor,
    candidate_phrases: List[str],
    encode_fn,
    top_k: int = 6,
    lambda_q: float = 0.8,
) -> List[str]:
    """
    Maximal Marginal Relevance:
    - favors semantic similarity to full question
    - penalizes redundancy among selected phrases
    """
    if not candidate_phrases:
        return []

    cand_embs = encode_fn(candidate_phrases)  # [N, D], normalized
    sims_to_q = torch.matmul(cand_embs, question_emb)  # [N]

    selected = []
    selected_idx = []

    remaining = list(range(len(candidate_phrases)))

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = -1e9

        for i in remaining:
            relevance = sims_to_q[i].item()

            if not selected_idx:
                diversity_penalty = 0.0
            else:
                sim_to_selected = torch.matmul(cand_embs[i].unsqueeze(0), cand_embs[selected_idx].T)
                diversity_penalty = sim_to_selected.max().item()

            score = lambda_q * relevance - (1 - lambda_q) * diversity_penalty

            # mild bonus for multiword phrases
            if len(candidate_phrases[i].split()) > 1:
                score += 0.03

            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(candidate_phrases[best_idx])
        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    return selected




# -----------------------------
# Embedding model
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" #"sentence-transformers/all-mpnet-base-v2"
#MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
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

def remove_single_words_in_phrases_ordered(keywords):
    multi = [k for k in keywords if len(k.split()) > 1]
    words_in_phrases = set()
    for phrase in multi:
        words_in_phrases.update(phrase.split())
    final = []
    for k in keywords:
        if len(k.split()) == 1 and k in words_in_phrases:
            continue
        if k not in final:
            final.append(k)
    return final

def extract_retrieval_keywords_semantic(
    text: str,
    encode_fn,
    top_k: int = 6,
) -> List[str]:
    candidates = extract_candidate_phrases(text)
    if not candidates:
        return []

    question_emb = encode_fn([text])[0]   # normalized
    keywords = mmr_select(
        question_emb=question_emb,
        candidate_phrases=candidates,
        encode_fn=encode_fn,
        top_k=top_k,
        lambda_q=0.8
    )
    return remove_single_words_in_phrases_ordered(keywords)


if __name__ == "__main__":
    question = "Why do we intentionally block the architecture from evaluating certain elements of the sequence during training and generation?"
    keywords = extract_retrieval_keywords_semantic(question, encode_texts, top_k=6)
    print("Question:", question)
    print("Extracted Keywords:", keywords)