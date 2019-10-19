from nltk import sent_tokenize, wordpunct_tokenize
from typing import Dict, List, Tuple
from .constants import PUNCT_SET
from .glove import Glove

import numpy as np

def cosine_similarity(a: np.float64, b: np.float64) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return -1
    return np.dot(a, b) / (norm_a * norm_b)

def _filter_words(text: str) -> List[str]:
    tokens = wordpunct_tokenize(text)
    words_filter = [word.lower() for word in tokens
                if set(word).isdisjoint(PUNCT_SET)]
    return words_filter

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
        return vector / len(words)

    def get_word_positions(self) -> Dict[str, List[int]]:
        words = _filter_words(self.text)
        word_positions = {}
        for i, word in enumerate(words):
            if word_positions.get(word) is None:
                word_positions[word] = [i]
            else:
                word_positions[word].append(i)
        return word_positions

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
    positions : List[Tuple[int]]
        List of indices where a given phrase is located. 
        Each index is represented as a Tuple where the first
        element is the first index the phrase appears in
        and the second element is the second index the phrase
        appears in. If a phrase is a unigram, a position Tuple
        is (position, position).
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
                parent: Document,
                glove: Glove) -> None:
        super().__init__(text, glove)
        self.parent = parent
        self.positions = self.__get_positions()
        self.window = self.__expand_window()
        self.similarity = cosine_similarity(parent.embedding, 
            self.embedding)
        self.theme_weight = None
        self.score = None
        self.rank = None

    def __str__(self) -> str:
        return self.text

    def set_theme_weight(self, 
                min_: float, 
                max_: float) -> None:
        # THIS SHOULD BE SET_THEME_EMBEDDING!!!!!
        diff = max_ - min_
        self.theme_weight = (self.similarity - min_) / diff

    def calc_pmi(self, phrase, candidates: int):
        """Calculates point-wise mutual information between
        one candidate phrase and another."""
        prob_phrase_one = len(self.positions) / candidates
        prob_phrase_two = len(phrase.positions) / candidates
        cooccur = 0
        for pos in phrase.positions:
            if self.window.get(pos[0]) or self.window.get(pos[1]):
                cooccur += 1
        prob_cooccur = cooccur / candidates
        return np.log(prob_cooccur / (prob_phrase_one * prob_phrase_two))

    def __get_positions(self) -> List[Tuple[int]]:
        """Gets positions a phrase is in."""
        parent_word_positions = self.parent.get_word_positions()
        phrase_split = self.text.lower().split(' ')
        positions = []
        if len(phrase_split) == 1:
            for word_pos in parent_word_positions[phrase_split[0]]:
                positions.append((word_pos, word_pos))
        else:
            phrase = {word: parent_word_positions[word] 
                    for word in phrase_split}
            len_phrase = len(phrase_split)
            for position in phrase[phrase_split[0]]:
                for i, word in enumerate(phrase_split[1:]):
                    pred_pos = position + i + 1
                    end_of_phrase = i + 2 == len_phrase
                    is_pred_pos = pred_pos in phrase[word]
                    if is_pred_pos and end_of_phrase:
                        positions.append((position, pred_pos))
        return positions

    def __expand_window(self) -> Dict[int, int]:
        """Returns dictionary of positions in a phrase's 
        adj. window."""
        window = {}
        phrase_len = len(self.parent.text.split(' '))
        for pos in self.positions:
            min_index = max(pos[0] - 5, 0)
            max_index = min(pos[1] + 6, phrase_len)
            indices = [i for i in range(min_index, max_index)]
            for i in indices:
                if window.get(i) is None:
                    window[i] = i
        return window