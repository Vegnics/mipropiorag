from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
#sentences = ['this may be a sample sentence', 'Each sentence is an example']
#sentences = ['verify that this may be a sample sentence', 'in this dataset each sentence is an example']
#sentences = ['why reading a sentence is important for understanding this data', 'in this dataset each sentence is an example']
sentences = ["Why must we prevent the model from looking at certain parts of the sequence while it is learning or generating text", 'Masked self-attention is used to ensure that the model doesn’t attend to some of the tokens in the input sequence during training or generation.']
#sentences = ["MLP: Multilayer Perceptron Attention attends over all the vectors Add positional encoding","Why must we prevent the model from looking at certain parts of the sequence while it is learning or generating text"]
#sentences = ["Why must we prevent the model from looking at certain parts of the sequence while it is learning or generating text","Desiderata of pos(.) : 1. It should output a unique encoding for each time-step (word’s position in a sentence) 2. Distance between any two time-steps should be consistent across sentences with different lengths.3. Our model should generalize to longer sentences without any efforts. Its values should be bounded. 4. It must be deterministic."]


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)


#print(model_output)
#print(model_output[0].shape,model_output[1].shape)
#print(encoded_input['attention_mask'].shape)


# Perform pooling
#sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

sentence_embeddings = model_output[1] ## Using the embbeddings from BertPooler
# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings.shape)
#print(sentence_embeddings.detach().numpy()[1])
#print(sentence_embeddings[1])
sim = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0),sentence_embeddings[1].unsqueeze(0),dim=-1)
dist = 1000*torch.square(sentence_embeddings[0]-sentence_embeddings[1]).mean()
print(sim,dist)