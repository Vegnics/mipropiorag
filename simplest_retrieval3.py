from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import csv
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np
import textwrap
import pymupdf

MODEL_NAME1 = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME2 = "sentence-transformers/all-mpnet-base-v2"
SLIDESPATH = "slides_dict_descriptions.json" #"slides_dict_3.json"
QUESTIONPATH = "HW1_questions.json"
#QUESTION = "When attempting to restore incomplete audio-visual streams, how is the backward noise-reduction step visually differentiated when it combines authentic input with artificially generated substitute data?"
#QUESTION = "Why must we prevent the model from looking at certain parts of the sequence while it is learning or generating text?"

tokenizer1 = AutoTokenizer.from_pretrained(MODEL_NAME1)
model1 = AutoModel.from_pretrained(MODEL_NAME1)
model1.eval()

tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME2)
model2 = AutoModel.from_pretrained(MODEL_NAME2)
model2.eval()

def emb_sim(emb1,emb2):
    sim = F.cosine_similarity(
                emb1.unsqueeze(0),
                emb2.unsqueeze(0),
                dim=-1
            ).item()
    return sim

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_texts(texts):
    """
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
    """
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
    return sentence_embeddings2 #torch.concat([sentence_embeddings1,sentence_embeddings2],dim=1)



with open(SLIDESPATH, "r", encoding="utf-8") as file:
    slidesdict = json.load(file)

with open(QUESTIONPATH, "r", encoding="utf-8") as file:
    questionsdict = json.load(file)

## Compute the embeddings for all the slides
#"""
sembeds = []
for i, (k, v) in tqdm(enumerate(slidesdict.items())):
    desc = v["slide_desc"]
    slide_embedding = encode_texts([desc])[0]
    sembeds.append(slide_embedding)
#"""

"""
tembeds = []
bembeds = []
ocrembeds = []
for i, (k, v) in tqdm(enumerate(slidesdict.items())):
    title = v["title"]
    body = v["body"]
    imgtxt = v["imgtext"]
    #title_embed = encode_texts([title])[0]
    body_embed = encode_texts([title+ " " + body])[0]
    imgtxt_embed = encode_texts([title + " " + imgtxt])[0]
    #tembeds.append(title_embed)
    bembeds.append(body_embed)
    ocrembeds.append(imgtxt_embed)
"""

IMAGEFOLDER ="slides_imgs"
slides = pymupdf.open('AI.pdf') ## Open slides PDF with pymupdf
answerspdf = pymupdf.open() ## PDF containing question / answer page
# Save results to CSV
data = [["Question","Slide number","Confidence","Content"]]
with open('raw_answers.csv', mode='w', newline='') as file:
    for qnum in range(200):
        # Encode question once
        QUESTION = questionsdict[qnum]["question"]
        ques_embed = encode_texts([QUESTION])[0]
        print(f"Answering Q-{qnum+1}:....")
        results = []
        confscores = []
        for i, (k, v) in tqdm(enumerate(slidesdict.items())):
            body = v["slide_desc"]
            #title = v["title"]
            #body = v["body"]
            #imgtxt = v["imgtext"]
            wbemd = 1.0
            wocr = 0.2
            #if len(imgtxt)<6:
            #    wocr = 0.0
            #text = title + " " + body + " "
            #tembd = tembeds[i]
            bemb = sembeds[i]
            #ocremb = ocrembeds[i]
            #emb_full = bemb+ocremb
            sim_full = wbemd*emb_sim(ques_embed,bemb)#+wocr*emb_sim(ques_embed,ocremb) #emb_sim(ques_embed,tembd)+emb_sim(ques_embed,bemb)+emb_sim(ques_embed,ocremb)
            sim = sim_full
            #results.append([int(k)+1, sim, title+"\n"+body])
            results.append([int(k)+1, sim,"\n"+body])
            confscores.append(sim) 
            #print(f"{i+1:3d} | page/key={k} | sim={sim:.4f}")
        # Sort best matches
        scores = torch.tensor(confscores, dtype=torch.float32)
        confidences = F.softmax(10 * scores, dim=0).tolist()
        confsum = sum(confidences)
        for k, res in enumerate(results):
            res[1] = confidences[k]
        results.sort(key=lambda x: x[1], reverse=True)
        #print(f"Question {QUESTION} \nPage: {results[0][0]} \nAns: {results[0][2]}\n Confidence: {results[0][1]}/{confsum}")
        print(f"Question {QUESTION} \nPage: {results[0][0]} \nAns: {results[0][2]}\n Confidence: {results[0][1]}/{confsum}")
        
        ## Generate the PDF with Questions And Answers
        respage = results[0][0]-1
        answerspdf.insert_pdf(slides, from_page=respage, to_page=respage)
        # get the newly inserted page
        new_page = answerspdf[-1]
        # add annotation
        rect = pymupdf.Rect(600, 50, 900, 250)
        annot = new_page.add_freetext_annot(
            rect,
            f"QUESTION: {QUESTION}",
            fontsize=18,
            text_color=(1, 0.3, 0),
            fill_color=(1, 1, 0.8),
        )
        annot.update()
        
        #simg = Image.open(os.path.join(IMAGEFOLDER,f'{results[0][0]:04}.jpg'))
        #simg = np.asarray(simg)
        #plt.figure(figsize=(12, 4))
        #plt.imshow(simg)
        #plt.axis("off")
        #wrap_question = "\n".join(textwrap.wrap(QUESTION, width=80))  # adjust width
        #plt.text(5, 10, f"Question: {wrap_question}", fontsize=12, color='black', bbox={'facecolor': 'white', 'pad': 10})
        #plt.show()
        
        #input("press enter...")
        data.append([QUESTION,results[0][0],results[0][1],results[0][2]])
    writer = csv.writer(file)
    writer.writerows(data)

answerspdf.save("answers_annotated.pdf")
    
"""
print(f"Question : {QUESTION}")
print("\nTop matches:")
for rank, (k, sim, v) in enumerate(results[:3], start=1):
    print(f"{rank:2d}. page/key={k} | sim={sim:.4f}")
    print(v)
    #print(v[:300])
    print("-" * 80)
input("Press Enter to continue.")
"""