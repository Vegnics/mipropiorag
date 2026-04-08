from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

FILEPATH = "slides_dict.json"
QUESTION = "Why must we prevent the model from looking at certain parts of the sequence while it is learning or generating text"
ANSWER = "Masked self-attention is used to ensure that the model doesn’t attend to some of the tokens in the input sequence during training or generation."

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def getSim(embeds):
    embeds = F.normalize(embeds, p=2, dim=1)
    sim = F.cosine_similarity(embeds[0].unsqueeze(0),embeds[1].unsqueeze(0),dim=-1)
    #sim = torch.square(embeds[0]-embeds[1]).mean()
    out = 1.0*float(sim.detach().numpy())
    #out = -10.0*torch.log(out+1e-8)
    return 1000.0*out

dictf = {}
similarities = []

with open(FILEPATH,'r') as file:
    slidesdict = json.load(file)

for i,(k,v) in enumerate(slidesdict.items()):
    sentences = [v,QUESTION]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    #sentence_embeddings = model_output[1] ## Using the embbeddings from BertPooler
    # Normalize embeddings
    #sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    sim = getSim(sentence_embeddings)
    print(f" {i+1} --> {sim:.03f}")
    if sim>=805:
        print(QUESTION,v) 
    #similarities.append()
    #similarities.append(1000*torch.square(sentence_embeddings[0]-sentence_embeddings[1]).mean())

"""
print(similarities)

sentences = [ANSWER,QUESTION]
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = model(**encoded_input)
sentence_embeddings = model_output[1] ## Using the embbeddings from BertPooler
# Normalize embeddings
#sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
print(getSim(sentence_embeddings))
"""