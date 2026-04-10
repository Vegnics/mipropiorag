from transformers import AutoTokenizer
from transformers import AutoModel
import torch
from torch.nn import functional as F

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
#MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" #"sentence-transformers/all-mpnet-base-v2"
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
        "What are the exact words written in the slide? What specific phrase or terminology is used to describe a concept or idea?",

    "entity_lookup":
        "What is the name of something. How a name or concept is mentioned in the slide?",

    "numeric":
        "What is the number, value, or quantity of <something>. How a quantity can be obtained?",

    "list":
        "How the slide presents several items, bullet points, or list elements?",

    "definition":
        "It describes something. What it is called, the meaning.",

    "property":
        "What something is like or what it has?. What specific characteristics it possesses?.",

    "comparison":
        "The things in the image are the same, different, similar?. How they are compared.",

    "structure":
        "What are the parts of something. How its components are organized.",

    "purpose_why":
        "Why something must be done?. What is the purpose to do something?. What is the reason behind a situation or fact?",

    "causal_reasoning":
        "What causes something or what happens because of it?. What is the effect or consequence of a situation?",

    "process_steps":
        "What are the steps or order to accomplish something?. What happens first, next, last? What is the sequence of events or stages in a process?",

    "mechanism_how":
        "How something works? What is the method, mechanism, or way that something happens or is done?",

    "visual_identification":
        "Can this scenario be visualized in a picture or diagram. It is not about a visual concept. Can something be identified in the image.",

    "visual_alignment":
        "How a picture and text relate to each other?. What in the image corresponds to something in the text?",

    "dataset_metric":
        "Is a  rule, score, or way used to measure something?. How a metric is calculated, or what data is used for a certain purpose?",

    "system_limitation":
        "What is the problem or weakness or limitation or drawbacks of something?",

    "compositional":
        "Do this need more than one piece of information?. How text, diagrams, and scenario can answer this together?",

    "example_case":
        "Does something describe a situation, or case, and asks which one it is?. An example or case description describe that phenomenom?"
}

def get_intention_scores(question):
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
	return sims


#question = "When attempting to restore incomplete audio-visual streams, how is the backward noise-reduction step visually differentiated when it combines authentic input with artificially generated substitute data?"
#question = "What is the initial mechanical setup required in the tangible mystery game before players can actively begin deducing the culprit's identity?"
#question = "Which specific 2022 academic manuscript involving latent representations is referenced regarding the creation of visual content from linguistic cues?"
#question = "When setting up the initial instructions for generating increasingly difficult gaming objectives, what specific format is used to supply supplementary contextual information to the system?"
#question = " In the section discussing logic challenges, which slide illustrates a scenario where a participant creates a hidden sequence of hues and the opponent attempts to figure it out based on numeric feedback?"
question = "Why do we intentionally block the architecture from evaluating certain elements of the sequence during training and generation?"

if __name__ == "__main__":
	sims = get_intention_scores(question)
	sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)

	sum_sims = sum(sorted_sims[i][1] for i in range(len(sorted_sims)))
	print("Question:", question)
	print("\nTop most similar question types:")
	for k, sim in sorted_sims:
		print(f"{k}: {sim:.4f}/{sum_sims:.4f}")