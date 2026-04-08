from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import csv

MODEL_NAME1 = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME2 = "sentence-transformers/all-mpnet-base-v2"
SLIDESPATH = "slides_dict_2.json"
QUESTIONPATH = "HW1_questions.json"
#QUESTION = "When attempting to restore incomplete audio-visual streams, how is the backward noise-reduction step visually differentiated when it combines authentic input with artificially generated substitute data?"
#QUESTION = "Why must we prevent the model from looking at certain parts of the sequence while it is learning or generating text?"

tokenizer1 = AutoTokenizer.from_pretrained(MODEL_NAME1)
model1 = AutoModel.from_pretrained(MODEL_NAME1)
model1.eval()

tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME2)
model2 = AutoModel.from_pretrained(MODEL_NAME2)
model2.eval()

#def mean_pooling(model_output, attention_mask):
#    token_embeddings = model_output.last_hidden_state
#    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

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
    return sentence_embeddings2 #torch.concat([sentence_embeddings1,sentence_embeddings2],dim=1)

with open(SLIDESPATH, "r", encoding="utf-8") as file:
    slidesdict = json.load(file)

with open(QUESTIONPATH, "r", encoding="utf-8") as file:
    questionsdict = json.load(file)

# Choose the question
qnum = 0


# Save results to CSV
data = [["Question","Slide number","Content"]]
with open('raw_answers.csv', mode='w', newline='') as file:
    for qnum in range(200):
        QUESTION = questionsdict[qnum]["question"]
        # Encode question once
        question_embedding = encode_texts([QUESTION])[0]
        print(f"Answering Q-{qnum+1}:....")
        results = []
        for i, (k, v) in tqdm(enumerate(slidesdict.items())):
            title = v["title"]
            body = v["body"]
            text = title + " " + body + " " #+ title
            #text = "masked self-attention is used to ensure that the model doesn attend to some of the tokens in the input sequence during training or generation. the architecture of vision transformer (vit) masked self-attention layer"
            slide_embedding = encode_texts([text])[0]
            sim_full = F.cosine_similarity(
                question_embedding.unsqueeze(0),
                slide_embedding.unsqueeze(0),
                dim=-1
            ).item()
            #sim = 0.65*sim_full+0.25*sim_body+0.1*sim_title
            sim = sim_full
            #results.append((k, sim, text))
            results.append((int(k)+1, sim, title+"\n"+body))
            #print(f"{i+1:3d} | page/key={k} | sim={sim:.4f}")
        # Sort best matches
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Question {QUESTION} \nPage: {results[0][0]} \nAns: {results[0][2]}")
        #input("press enter...")
        data.append([QUESTION,results[0][0],results[0][2]])
    writer = csv.writer(file)
    writer.writerows(data)

    
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