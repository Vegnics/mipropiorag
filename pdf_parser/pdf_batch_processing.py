from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import csv
import re
from lang_utils.raw_text_ops import clean_slide_text,clean_page_text
import pymupdf


def clean_pdf_text(text: str) -> str:
    # remove hyphenation across line breaks: "learn-\ning" -> "learning"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # convert single newlines inside paragraphs into spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # keep paragraph boundaries for double newlines
    text = re.sub(r'\n{2,}', '\n\n', text)

    # collapse repeated spaces
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()

def split_into_sentences(text: str):
    # simple sentence splitter
    # works reasonably well for papers, though not perfect for abbreviations
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_sentences(sentences, max_chars=500, min_chars=120):
    chunks = []
    cur = []

    for sent in sentences:
        #
        # sent = lemmatize_clean(sent)
        tentative = " ".join(cur + [sent]).strip()
        if len(tentative) <= max_chars:
            cur.append(sent)
        else:
            if cur:
                chunks.append(" ".join(cur).strip())
            cur = [sent]

    if cur:
        chunks.append(" ".join(cur).strip())

    # optional: merge very short tail chunks
    merged = []
    for ch in chunks:
        if merged and len(ch) < min_chars:
            merged[-1] += " " + ch
        else:
            merged.append(ch)

    return merged

def pdf_to_short_chunks(text, max_chars=500, min_chars=120):
    sentences = split_into_sentences(text)
    chunks = chunk_sentences(sentences, max_chars=max_chars, min_chars=min_chars)
    return chunks

# example
chunks = pdf_to_short_chunks("paper.pdf", max_chars=400, min_chars=100)

#MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
"""
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SLIDESPATH = "slides_dict.json"
QUESTIONPATH = "HW1_questions.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_texts(texts):
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings
"""

MODEL_NAME1 = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME2 = "sentence-transformers/all-mpnet-base-v2"

tokenizer1 = AutoTokenizer.from_pretrained(MODEL_NAME1)
model1 = AutoModel.from_pretrained(MODEL_NAME1)
model1.eval()

tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME2)
model2 = AutoModel.from_pretrained(MODEL_NAME2)
model2.eval()

###########################
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_texts(texts):
    encoded_input = tokenizer1(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model1(**encoded_input)
    sentence_embeddings1 = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings1 = F.normalize(sentence_embeddings1, p=2, dim=1)

    encoded_input = tokenizer2(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model2(**encoded_input)
    sentence_embeddings2 = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings2 = F.normalize(sentence_embeddings2, p=2, dim=1)
    #return torch.concat([sentence_embeddings1,sentence_embeddings2],dim=1)
    return sentence_embeddings2

#encode_texts([QUESTION])[0]
pdf_name = "papers/Attention Is All You Need.pdf"
doc = pymupdf.open(pdf_name)

slides_dict = {}
print(doc.page_count)
text_chunks = []
for k in range(doc.page_count):
    page = doc.load_page(k)
    text = page.get_text(sort=True)
    text = clean_page_text(text)
    slides_dict[k] = {}
    slides_dict[k]["body"] = text
    print(f"Page {k+1}: \n Body")#: {text}")
    print(len(text))
    chunks = pdf_to_short_chunks(text, max_chars=600, min_chars=200)
    for chunk in chunks:
        print("--> ",chunk,"\n")
    """
    for i in range(0,len(text),600):
        last = min(i+600,len(text))
        text_chunks.append(text[i:last])
    """
    text_chunks.extend(chunks)
    #print(text_chunks)
    input("Press Enter to continue..")
    print("\n\n")

chunks_dict = {}
for chunk in text_chunks:
    chunk_emb = encode_texts([chunk])[0]
    chunks_dict[chunk]=chunk_emb.squeeze().detach().numpy().tolist()

with open("embeds_attention.json",'w') as f:
    json.dump(chunks_dict,f)