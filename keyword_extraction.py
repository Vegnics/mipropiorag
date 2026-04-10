import spacy
import re


nlp = spacy.load("en_core_web_sm")

def clean_text(s: str):
    # keep letters, numbers, and spaces
    s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def remove_single_words_in_phrases_ordered(keywords):
    multi = [k for k in keywords if len(k.split()) > 1]
    words_in_phrases = set()
    for phrase in multi:
        words_in_phrases.update(phrase.split())
    final = []
    for k in keywords:
        if len(k.split()) == 1 and k in words_in_phrases:
            continue
        if k not in final:
            final.append(k)
    return final

def extract_retrieval_keywords(text: str):
    doc = nlp(text)
    keywords = []
    # 1) Noun chunks (keep original text, not lemma)
    for chunk in doc.noun_chunks:
        # clean chunk (remove stopwords at edges)
        tokens = [t.text for t in chunk if not t.is_stop and not t.is_punct]
        if tokens:
            phrase = " ".join(tokens)
            keywords.append(phrase.lower())
    # 2) Important adjectives (e.g., "different")
    for token in doc:
        if token.pos_ == "ADJ" and not token.is_stop:
            keywords.append(token.text.lower())
    # 3) Verb + object (e.g., "share architecture")
    for token in doc:
        if token.pos_ == "VERB" and not token.is_stop:
            for child in token.children:
                if child.dep_ in {"dobj", "attr", "pobj"}:
                    phrase = f"{token.lemma_} {child.text}".lower()
                    keywords.append(phrase)
    # 4) Deduplicate + clean
    cleaned = []
    seen = set()
    for k in keywords:
        k = clean_text(k)
        if k and k not in seen:
            seen.add(k)
            cleaned.append(k)
    return remove_single_words_in_phrases_ordered(cleaned)


if __name__ == "__main__":
    question = "When charting the cumulative success rates of different self-assessment methodologies for binary categorization, exactly how many distinct challenges were analyzed in the provided graph?"
    print(extract_retrieval_keywords(question))