## To do:
## 1) Write function to remove illegal candidate phrasers.
## 2) Write function to de-duplicate final candidate list.

import numpy as np
import spacy
import en_core_web_sm
import os

from nltk import sent_tokenize, wordpunct_tokenize
from typing import Dict, List
from .glove import Glove
from .docs import Document, Phrase

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

    def extract_candidates(self, 
            ents_to_ignore: List[str]=['DATE', 'TIME', 
            'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 
            'CARDINAL']) -> None:
        """Extracts candidate phrases from the text. Sets
        `candidates` attributes to a list of Phrase objects.
 
        Parameters
        ----------
        ents_to_ignore : List[str], optional 
            (List[str] = ['DATE', 'TIME', 
            'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 
            'CARDINAL'])
            Named entities to ignore during the candidate
            selection process.
        """ 

        sentences = sent_tokenize(self.doc.text)
        candidates = {}
        for sentence in sentences:
            doc = NLP(sentence)
            for ent in doc.ents:
                if ent.label_ not in ents_to_ignore:
                    if candidates.get(ent.text, None) is None:
                        candidates[ent.text] = Phrase(ent.text, 
                            self.glove, self.doc)
            for chunk in doc.noun_chunks:
                if candidates.get(chunk.text, None) is None:
                    candidates[chunk.text] = Phrase(chunk.text,
                        self.glove, self.doc)
        self.candidates = list(candidates.values())

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

        return sorted_candidates[:top_n - 1]