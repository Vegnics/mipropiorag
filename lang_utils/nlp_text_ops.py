import spacy

nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

def lemmatize_clean(text: str) -> str:
    doc = nlp(text)
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(lemmas)