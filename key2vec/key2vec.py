import numpy as np
import spacy
import en_core_web_sm
import os

from nltk import sent_tokenize, wordpunct_tokenize
from typing import Dict, List
from .utils import read_glove

NLP = en_core_web_sm.load()
ENTS_TO_IGNORE = ['DATE', 'TIME',
                'PERCENT', 'MONEY',
                'QUANTITY', 'ORDINAL',
                'CARDINAL']

class Key2Vec(object):

    """Implementation of Key2Vec.

    Parameters
    ----------
    text : str, required
        The text to extract the top keyphrases from.
    dim : int, required
        Embedding size for the text, and keyphrases. Must
        be 50, 100, 200, or 300.
    top_n : int, optional (default = 10)
        Number of top keyphrases to return after processing
        the text. Must be at least 1.
        **Note**: It may make more sense to pass top_n
        to a function like `extract`.
    """

    def __init__(self,
        text: str,
        path: str,
        dim: int,
        top_n: int=10) -> None:
        
        self.text = text
        self.path = path
        self.dim = dim

        if top_n < 1:
            raise ValueError('`top_n` must be greater than 1.')
        self.top_n = top_n
        self.glove = read_glove(path)

    def extract_candidates(self):
        sentences = sent_tokenize(self.text)
        candidates = {}
        for sentence in sentences:
            doc = NLP(sentence)
            for ent in doc.ents:
                if ent.label_ not in ENTS_TO_IGNORE:
                    if candidates.get(ent.text, None) is None:
                        candidates[ent.text] = [ent]
                    else:
                        candidates[ent.text].append(ent)
            for chunk in doc.noun_chunks:
                if candidates.get(chunk.text, None) is None:
                    candidates[chunk.text] = [chunk]
                else:
                    candidates[chunk.text].append(chunk)
        return candidates

    def clean_candidates(self):
        pass

    def embed_text(self):
        return self.embed_doc(self.text)

    def embed_candidates(self, candidates: List[str]):
        candidate_vectors = {c: None for c in candidates}
        for c in candidates:
            if candidate_vectors.get(c, None) is None:
                candidate_vectors[c] = self.embed_doc(c)
        return candidate_vectors

    def embed_doc(self, doc: str):
        words = wordpunct_tokenize(doc.lower())
        vector = np.zeros(self.dim)
        for i, word in enumerate(words):
            if self.glove.get(word, None) is None:
                vector += np.zeros(self.dim)
            else:
                vector += self.glove[word]
        return vector / (i + 1)

    def rank_candidates(self, candidates: List[str]):
        text_embed = self.embed_text()
        candidates_embed = self.embed_candidates(candidates)
        candidate_scores = {c: {'sim': None, 
                            'score': None, 
                            'rank': None} for c in candidates}

        for candidate in candidates:
            similarity = self.cosine_similarity(text_embed, 
                candidates_embed[candidate])
            candidate_scores[candidate]['sim'] = similarity

        max_sim = max([candidate_scores[key]['sim'] for 
            key in candidates])
        min_sim = min([candidate_scores[key]['sim'] for 
            key in candidates])
        diff = max_sim - min_sim

        for c in candidates:
            score = (candidate_scores[c]['sim'] - min_sim) / diff
            candidate_scores[c]['score'] = score

        return candidate_scores

    def cosine_similarity(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            return -1
        return np.dot(a, b) / (norm_a * norm_b)