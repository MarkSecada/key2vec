import numpy as np
import spacy
import string
import en_core_web_sm
import os

from nltk import sent_tokenize, wordpunct_tokenize
from typing import Dict, List
from .cleaner import Cleaner
from .constants import ENTS_TO_IGNORE, STOPWORDS, PUNCT_SET
from .docs import Document, Phrase
from .glove import Glove
from .phrase_graph import PhraseNode, PhraseGraph

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
    candidate_graph : PhraseGraph
        Bidrectional graph of all candidate phrases
    """

    def __init__(self,
        text: str,
        glove: Glove) -> None:
        
        self.doc = Document(text, glove)
        self.glove = glove
        self.candidates = []
        self.candidate_graph = None

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
            not_punct = set(text).isdisjoint(PUNCT_SET)
            is_stopword = text in STOPWORDS
            in_candidates = candidates.get(text) is not None
            not_empty = text != ''
            keep = (not_punct
                and not_empty
                and not (is_stopword or in_candidates))
            if keep:
                try:
                    candidates[text] = Phrase(text, self.doc, 
                        self.glove)
                except KeyError:
                    next
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
                try:
                    candidates[cleaned_text] = Phrase(cleaned_text, self.doc,
                        self.glove)
                except KeyError:
                    next
        return candidates

    def __extract_noun_chunks(self, doc, candidates):
        for chunk in doc.noun_chunks:
            cleaned_text = Cleaner(chunk).transform_text()
            not_empty = cleaned_text != ''
            if candidates.get(cleaned_text) is None and not_empty:
                try:
                    candidates[cleaned_text] = Phrase(cleaned_text, 
                        self.doc, self.glove)
                except KeyError:
                    next
        return candidates

    def set_theme_weights(self) -> List[Phrase]:
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
        max_ = max([c.similarity for c in self.candidates])
        min_ = min([c.similarity for c in self.candidates])

        for c in self.candidates:
            c.set_theme_weight(min_, max_)

    def build_candidate_graph(self) -> None:
        """Builds bidirectional graph of candidates."""

        if self.candidates == []:
            return

        candidate_graph = PhraseGraph(self.candidates)
        for candidate in self.candidates:
            candidate_graph.add_node(candidate)

        nodes = len(self.candidates)

        for node in candidate_graph.nodes:
            for other in candidate_graph.nodes:
                if node != other:
                    candidate_graph.nodes[node].add_neighbor(
                        candidate_graph.nodes[other], nodes)
        self.candidate_graph = candidate_graph

    def page_rank_candidates(self, top_n: int=10) -> List[Phrase]:
        """Page Ranks candidate phrases."""
        if self.candidate_graph is None:
            return

        for node in self.candidate_graph.nodes.values():
            theme = node.phrase.theme_weight
            d = 0.85
            weights = []
            neighbors = list(node.adj_nodes.keys())
            for neighbor in neighbors:
                out = node.adj_nodes[neighbor].incoming_edges
                weights.append(node.adj_weights[neighbor] / out)
            score = theme * (1 - d) + d * sum(weights)
            node.phrase.score = score

        sorted_candidates = sorted(self.candidates, 
            key=lambda x: x.score)[::-1]

        for i, c in enumerate(sorted_candidates):
            c.rank = i + 1

        return sorted_candidates[:top_n]