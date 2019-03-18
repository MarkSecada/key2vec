from nltk import sent_tokenize, wordpunct_tokenize
from typing import Dict, List
from .glove import Glove

import numpy as np

class Document(object):
    """Document to be embedded. May be a word, a sentence, etc.

    Parameters
    ----------
    text : str, required
        The text to be embedded
    glove : Glove, required
        GloVe embeddings

    Attributes
    ----------
    text : str
    dim : int
        Dimension of GloVe embeddings.
    embedding : np.float64
        Document embedding built from average of GloVe embeddings.
    """

    def __init__(self, 
                text: str, 
                glove: Glove) -> None:
        self.text = text
        self.dim = glove.dim
        self.embedding = self.__embed_document(glove.embeddings)

    def __embed_document(self, 
                embeddings: Dict[str, np.float64]) -> np.float64:
        words = wordpunct_tokenize(self.text.lower())
        vector = np.zeros(self.dim)
        for i, word in enumerate(words):
            if embeddings.get(word, None) is None:
                vector += np.zeros(self.dim)
            else:
                vector += embeddings[word]
        return vector / (i + 1)

class Phrase(Document):
    """Phrase to be embedded. Inherits from Document object.

    Parameters
    ----------
    text : str, required
        The text to be embedded
    glove : Glove, required
        GloVe embeddings
    parent : Document, required
        Document where the Phrase is from

    Attributes
    ----------
    text : str
    dim : int
    embedding : np.float64
    parent : Document
    similarity : float
        Cosine similarity between the parent document and the phrase.
    score : float, None
        Min/Max scaling of the cosine similarity in relation to the
        other candidate keyphrases.
    rank : int, None
        Phrase ranking with respect to the score in descending order.
    """

    def __init__(self, 
                text: str, 
                glove: Glove, 
                parent: Document) -> None:
        super().__init__(text, glove)
        self.parent = parent
        self.similarity = self.__cosine_similarity(parent.embedding,
            self.embedding)
        self.score = None
        self.rank = None

    def set_score(self, 
                min_: float, 
                max_: float) -> None:
        diff = max_ - min_
        self.score = (self.similarity - min_) / diff

    def __cosine_similarity(self, 
                a: np.float64, 
                b: np.float64) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            return -1
        return np.dot(a, b) / (norm_a * norm_b)