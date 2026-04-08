from pdf2image import convert_from_path
import os
import tqdm
import json
import re
import pymupdf
import unicodedata
from pypdf import PdfReader

import sys
import os
from lang_utils.raw_text_ops import clean_slide_text,clean_page_text
from pdf_parser.pdf_finder import find_and_download_arxiv

from PIL import Image
import io
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
#from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import AutoProcessor,AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
from visual.ocr import OCR_Reader

from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


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
####################

def get_caption(image_np,mod0,proc0,mod1,proc1,text):
    image = Image.fromarray(image_np)
    inputs = proc0(images=image,return_tensors="pt").to(device)
    #inputs["input_ids"]=None
    with torch.no_grad():
        out = mod0.generate(**inputs, max_new_tokens=40)
    caption = proc0.decode(out[0], skip_special_tokens=True)
    ## Block diagram
    if "block" in caption and "diagram" in caption or True:
        #question = f"According to {text}.According to {text}. What is the block diagram explaining?"
        question = f"AI Lecture. Slide Title: {text} The block diagram explains .."
        inputs = proc1(images=image,text=question, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mod1.generate(**inputs, max_new_tokens=40)
        caption = "Block Diagram: "+ proc1.decode(out[0], skip_special_tokens=True)
        print(caption)
        plt.imshow(img_np)
        plt.show()
    return caption

def split_reference(ref: str):
    parts = [p.strip() for p in ref.split(",")]
    if len(parts) < 3:
        return None
    return {
        "title": ", ".join(parts[:-2]),   # handles commas inside title
        "conference": parts[-2],
        "year": int(parts[-1])
    }


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



def dict_embedding_retrieve(dict_emb,src_emb):
    sims = []
    text = []
    for k,v in dict_emb.items():
        emb = torch.tensor(v)
        sim = F.cosine_similarity(
                emb.unsqueeze(0),
                src_emb.unsqueeze(0),
                dim=-1
            ).item()
        sims.append(sim)
        text.append(k)
    res = sorted(zip(sims,text),key=lambda x:x[0],reverse=True)
    res = list(res)
    return res[0][1]
        
def text_from_blocks(blocks):
    tblocks = []
    for b in blocks:
        full_text = ""
        if b["type"] == 0:
            origin = None
            for k in range(len(b["lines"])):
                line = b["lines"][k]
                spans = line["spans"]
                if len(spans) == 0:
                    continue
                if origin is None:
                    origin = spans[0]["origin"]
                text = "".join(span["text"] for span in spans)
                full_text += text + " "
            full_text = full_text.strip()
            if origin is not None and full_text:
                tblocks.append({"text": full_text, "origin": origin})
    tblocks = sorted(tblocks, key=lambda x: (x["origin"][1], x["origin"][0]))
    full_text = ""
    for b in tblocks:
        btext = b["text"]
        full_text += f"{btext}\n"
    return full_text
  

"""
proc0 = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model0 = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base").to(device)


proc1 = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model1 = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
"""


#processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
#model = BlipForQuestionAnswering.from_pretrained(
#    "Salesforce/blip-vqa-base",
#    torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

doc = pymupdf.open('AI.pdf')
ocr_reader = OCR_Reader()
# getting a specific page from the pdf file

slides_dict = {}
#for k in range(len(reader.pages)):
print(doc.page_count)
for k in range(doc.page_count):
    page = doc.load_page(k)
    w = page.rect.x1
    h = page.rect.y1
    page_json = page.get_text(option="dict")
    blocks = page_json["blocks"]
    ## Text blocks list
    text = text_from_blocks(blocks)

    imgblocks = [b for b in blocks if b["type"] == 1]
    imgtext = ""
    # first image block
    for b in imgblocks:
        # decode image bytes to PIL, then to numpy
        img = Image.open(io.BytesIO(b["image"])).convert("RGB")
        img = np.array(img)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        imgtext += ocr_reader.image2text_ocr(img) # Cleaned OCR
        imgtext += "\n"
    imgtext = clean_page_text(imgtext)
    #text = page.get_text(sort=True)
    title,text = clean_slide_text(text)
    slides_dict[k] = {}
    slides_dict[k]["title"] = title
    slides_dict[k]["body"] = text
    slides_dict[k]["imgtext"] = imgtext
    #"""
    #src_emb = encode_texts([title+" "+text+" "+title])[0]
    slides_dict[k]["reference"] = None
    print(f"Page {k+1}:  \n",f"\tTitle: {title}\n",f"\tBody: {text}\n",f"\tImage Content:{imgtext}\n\n")
    reference = extract_source_block(text)
    if reference is not None:
        """
        slides_dict[k]["reference"] = reference
        if False and slides_dict[k]["reference"]["title"] == "Attention is All You Need": #re.search(slides_dict[k]["reference"]["title"],"Attention Is All You Need"):
            with open("embeds_attention.json",'r') as f:
                dict_pdf = json.load(f)
            retrieved = dict_embedding_retrieve(dict_pdf,src_emb)
            print(f"Retrieved: {retrieved}")
            input("Press Enter ....")
        #find_and_download_arxiv(reference["title"])
        """
        pass
        #print(f"REFERENCE: {reference}")
    #"""
    #input("Press Enter ....")
    """
    page_json = page.get_text(option="dict")
    blocks = page_json["blocks"]
    print(blocks)
    imgblocks = [b for b in blocks if b["type"] == 1]
    # first image block
    if len(imgblocks)>0:
        b = imgblocks[0]
        # decode image bytes to PIL, then to numpy
        img = Image.open(io.BytesIO(b["image"])).convert("RGB")
        img_np = np.array(img)
        caption = get_caption(img_np,model0,proc0,model1,proc1,title)
        print(f"Image Content: {caption}")
    
    """
    

with open("slides_dict_3.json",'w') as f:
    json.dump(slides_dict,f)

# Convert PDF to a list of PIL Image objects
#images = convert_from_path('AI.pdf', dpi=300)

# Save each page as a JPEG file
#for i, image in tqdm.tqdm(enumerate(images)):
#    image.save(os.path.join("slides_imgs",f'{i:04}.jpg'), 'JPEG')