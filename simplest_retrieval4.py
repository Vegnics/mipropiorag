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
SLIDESPATH = "full_slides_dict.json" #"slides_dict_3.json"
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

from collections import defaultdict
from tqdm import tqdm



def weighted_rrf(rank_lists, weights, k=60):
	fused_scores = defaultdict(float)
	page_meta = {}

	for rank_list, w in zip(rank_lists, weights):
		for rank, item in enumerate(rank_list, start=1):
			page = item[0]
			fused_scores[page] += w * (1.0 / (k + rank))
			if page not in page_meta:
				page_meta[page] = item

	fused_results = []
	for page, score in fused_scores.items():
		meta = page_meta[page]
		fused_results.append([page, score, meta[2]])

	fused_results.sort(key=lambda x: x[1], reverse=True)
	return fused_results

def reciprocal_rank_fusion(rank_lists, k=60):
	"""
	rank_lists: list of ranked lists
		each ranked list is like [[page, score, text], ...] already sorted desc
	returns:
		fused_results: [[page, fused_score, extra], ...] sorted desc
	"""
	fused_scores = defaultdict(float)
	page_meta = {}

	for rank_list in rank_lists:
		for rank, item in enumerate(rank_list, start=1):
			page = item[0]
			fused_scores[page] += 1.0 / (k + rank)

			# keep one copy of metadata
			if page not in page_meta:
				page_meta[page] = item

	fused_results = []
	for page, score in fused_scores.items():
		meta = page_meta[page]
		fused_results.append([page, score, meta[2]])

	fused_results.sort(key=lambda x: x[1], reverse=True)
	return fused_results


with open(SLIDESPATH, "r", encoding="utf-8") as file:
	slidesdict = json.load(file)

with open(QUESTIONPATH, "r", encoding="utf-8") as file:
	questionsdict = json.load(file)

## Compute the embeddings for all the slides
"""
sembeds = []
for i, (k, v) in tqdm(enumerate(slidesdict.items())):
	desc = v["slide_desc"]
	slide_embedding = encode_texts([desc])[0]
	sembeds.append(slide_embedding)
"""

#"""
bembeds = []
dembeds = []
for i, (k, v) in tqdm(enumerate(slidesdict.items())):
	title = v["title"]
	body = v["body"]
	imgtxt = v["imgtext"]
	desc = v["slide_desc"]
	body_embed = encode_texts([title + " "+ body+ " " + imgtxt])[0]
	desc_embed = encode_texts([desc])[0]
	#tembeds.append(title_embed)
	bembeds.append(body_embed)
	dembeds.append(desc_embed)
#"""

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
		results_txt = []
		results_desc = []
		results = []
		#confscores = []
		for i, (k, v) in tqdm(enumerate(slidesdict.items())):
			body = v["slide_desc"]
			title = v["title"]
			body = v["body"]
			imgtxt = v["imgtext"]
			wdesc = 0.4
			wtxt = 0.6
			if len(body+imgtxt)<10:
				wdesc = 1.0
				wtxt = 0.0
			bemb = bembeds[i]
			demb = dembeds[i]
			sim_txt = emb_sim(ques_embed,bemb)
			sim_desc = emb_sim(ques_embed,demb)
			#sim = max(0.9*sim_txt*wtxt,0.8*sim_desc,0.6*sim_desc+0.4*sim_txt*wtxt)
			sim = wdesc*sim_desc + wtxt*sim_txt
			results_desc.append([int(k)+1, sim_desc,"\n"+body])
			results_txt.append([int(k)+1, sim_txt,"\n"+body])
			results.append([int(k)+1, sim,"\n"+body])
			#print(f"{i+1:3d} | page/key={k} | sim={sim:.4f}")
		# Sort best matches
		results_desc.sort(key=lambda x: x[1], reverse=True)
		results_txt.sort(key=lambda x: x[1], reverse=True)
		results.sort(key=lambda x: x[1], reverse=True)
		top_res_desc = results_desc[:40]
		top_res_txt = results_txt[:40]

		final_rank = results #weighted_rrf([top_res_desc, top_res_txt],weights=[0.3,0.7], k=20)#reciprocal_rank_fusion([top_res_desc, top_res_txt], k=30)

		#print(f"Question {QUESTION} \nPage: {results[0][0]} \nAns: {results[0][2]}\n Confidence: {results[0][1]}/{confsum}")
		print(f"Question {QUESTION} \nPage: {final_rank[0][0]} \nAns: {final_rank[0][2]}\n Confidence: {final_rank[0][1]}")
		
		## Generate the PDF with Questions And Answers
		respage = final_rank[0][0]-1
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
		data.append([QUESTION,final_rank[0][0],final_rank[0][1],final_rank[0][2]])
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