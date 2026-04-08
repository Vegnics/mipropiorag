from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import csv
import re
from raw_text_ops import clean_slide_text,clean_page_text,lemmatize_text,lemmatize_clean
import pymupdf
import os


doc = pymupdf.open('AI.pdf')
OUTFOLDER ="slides_imgs"
os.makedirs(OUTFOLDER,exist_ok=True)
for k in range(doc.page_count):
    page = doc.load_page(k)
    pixmap = page.get_pixmap(dpi=200)
    pixmap.save(os.path.join(OUTFOLDER,f'{k+1:04}.jpg'))
