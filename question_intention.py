from transformers import AutoTokenizer
from transformers import AutoModel
import torch
from torch.nn import functional as F

#MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" #"sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

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
	encoded_input = tokenizer(
		texts,
		padding=True,
		truncation=True,
		return_tensors="pt"
	)
	with torch.no_grad():
		model_output = model(**encoded_input)
	sentence_embeddings2 = mean_pooling(model_output, encoded_input["attention_mask"])
	sentence_embeddings2 = F.normalize(sentence_embeddings2, p=2, dim=1)
	return sentence_embeddings2 

QUESTION_DESCRIPTIONS = {
    "phrase_lookup":
        "The question asks for the exact words written in the slide.",

    "entity_lookup":
        "The question asks for the name of something.",

    "numeric":
        "The question asks for a number.",

    "list":
        "The question asks for several items.",

    "definition":
        "The question describes something and asks what it is called.",

    "property":
        "The question asks what something is like or what it has.",

    "comparison":
        "The question asks if things are the same or different.",

    "structure":
        "The question asks about parts of something.",

    "purpose_why":
        "The question asks why something is done.",

    "causal_reasoning":
        "The question asks what causes something or what happens because of it.",

    "process_steps":
        "The question asks about steps or order.",

    "mechanism_how":
        "The question asks how something works.",

    "visual_identification":
        "The question asks about something shown in a picture or diagram.",

    "visual_alignment":
        "The question asks how a picture and text relate to each other.",

    "dataset_metric":
        "The question asks about a rule, score, or way to measure something.",

    "system_limitation":
        "The question asks about a problem or weakness.",

    "compositional":
        "The question needs more than one piece of information.",

    "example_case":
        "The question describes a situation and asks which one it is."
}


#question = "When attempting to restore incomplete audio-visual streams, how is the backward noise-reduction step visually differentiated when it combines authentic input with artificially generated substitute data?"
#question = "What is the initial mechanical setup required in the tangible mystery game before players can actively begin deducing the culprit's identity?"
question = "Which specific 2022 academic manuscript involving latent representations is referenced regarding the creation of visual content from linguistic cues?"
#question = "When setting up the initial instructions for generating increasingly difficult gaming objectives, what specific format is used to supply supplementary contextual information to the system?"
#question = " In the section discussing logic challenges, which slide illustrates a scenario where a participant creates a hidden sequence of hues and the opponent attempts to figure it out based on numeric feedback?"

sims = {}

for k,v in QUESTION_DESCRIPTIONS.items():
    emb1 = encode_texts([question])[0]
    emb2 = encode_texts([v])[0]
    sim = emb_sim(emb1,emb2)
    sims[k] = sim*10.0

## Softmax over similarities
sim_list = list(sims.values())
exp_sims = [torch.exp(torch.tensor(sim)) for sim in sim_list]
sum_exp_sims = sum(exp_sims)
for k, sim in sims.items():
    sims[k] = torch.exp(torch.tensor(sim)) / sum_exp_sims
sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)

sum_sims = sum(sorted_sims[i][1] for i in range(len(sorted_sims)))
print("Question:", question)
print("\nTop 5 most similar question types:")
for k, sim in sorted_sims[:5]:
    print(f"{k}: {sim:.4f}/{sum_sims:.4f}")