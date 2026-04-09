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
