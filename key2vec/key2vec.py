## To do:
## 1) Need to write out a way to cleaned entities, nouns, etc.
## 2) Write function to de-duplicate final candidate list.

import numpy as np
import spacy
import string
import en_core_web_sm
import os

from nltk import sent_tokenize, wordpunct_tokenize
from typing import Dict, List
from .cleaner import Cleaner
from .constants import ENTS_TO_IGNORE, STOPWORDS
from .docs import Document, Phrase
from .glove import Glove

NLP = en_core_web_sm.load()

class Key2Vec(object):
    """Implementation of Key2Vec.

    Parameters
    ----------
    text : str, required
        The text to extract the top keyphrases from.
    glove : Glove
        GloVe vectors.

    Attributes
    ----------
    text : Document
        Document object of the `text` parameter.
    glove : Glove
    candidates : List[Phrase]
        List of candidate keyphrases. Initialized as an empty list.
    """

    def __init__(self,
        text: str,
        glove: Glove) -> None:
        
        self.doc = Document(text, glove)
        self.glove = glove
        self.candidates = []

    def extract_candidates(self):
        """Extracts candidate phrases from the text. Sets
        `candidates` attributes to a list of Phrase objects.
        """ 
        sentences = sent_tokenize(self.doc.text)
        candidates = {}
        for sentence in sentences:
            doc = NLP(sentence)
            candidates = self.__extract_tokens(doc, candidates)
            candidates = self.__extract_entities(doc, candidates)
            candidates = self.__extract_noun_chunks(doc, candidates)
        self.candidates = list(candidates.values())

    def __extract_tokens(self, doc, candidates):
        for token in doc:
            text = token.text.lower()
            is_punct = text in string.punctuation
            is_dash = text == '-'
            is_stopword = text in STOPWORDS
            in_candidates = candidates.get(text) is not None
            if is_dash and not in_candidates:
                candidates[text] = Phrase(token.text, self.glove,
                    self.doc)
            elif not (is_dash or is_stopword or in_candidates):
                candidates[text] = Phrase(token.text, self.glove,
                    self.doc)
            else:
                pass
        return candidates

    def __extract_entities(self, doc, candidates):
        for ent in doc.ents:
            cleaned_text = Cleaner(ent).transform_text()
            is_ent_to_ignore = ent.label_ in ENTS_TO_IGNORE
            in_candidates = candidates.get(cleaned_text) is not None
            not_empty = cleaned_text != ''
            if not (is_ent_to_ignore or in_candidates) and not_empty:
                candidates[cleaned_text] = Phrase(ent.text, 
                    self.glove, self.doc)
        return candidates

    def __extract_noun_chunks(self, doc, candidates):
        for chunk in doc.noun_chunks:
            cleaned_text = Cleaner(chunk).transform_text()
            not_empty = cleaned_text != ''
            if candidates.get(cleaned_text) is None and not_empty:
                candidates[cleaned_text] = Phrase(chunk.text, 
                    self.glove, self.doc)
        return candidates

    def rank_candidates(self, top_n: int=10) -> List[Phrase]:
        """Ranks candidate keyphrases.

        Parameters
        ----------
        top_n : int, optional (int = 10)
            How many top keyphrases to return.

        Returns
        -------
        sorted_candidates : List[Phrase]
            Sorted list of candidates in reverse order. Returns `top_n`
            Phrase objects.
        """

        if top_n < 1:
            raise ValueError('`top_n` must be greater than 1.')

        max_ = max([c.similarity for c in self.candidates])
        min_ = min([c.similarity for c in self.candidates])

        for c in self.candidates:
            c.set_score(min_, max_)

        sorted_candidates = sorted(self.candidates, 
            key=lambda x: x.score)[::-1]

        for i, c in enumerate(sorted_candidates):
            c.rank = i + 1

        return sorted_candidates[:top_n]