from .docs import Document, Phrase, cosine_similarity
from typing import List

class PhraseNode(object):
    """Node in Phrase Graph."""

    def __init__(self, phrase: Phrase):
        self.key = phrase.text
        self.phrase = phrase
        self.incoming_edges = 0
        self.adj_nodes = {}
        self.adj_weights = {}

    def __repr__(self):
        return str(self.key)

    def __lt__(self, other):
        return self.key < other.key

    def add_neighbor(self, neighbor, candidates, weight=0):
        if neighbor is None or weight is None:
            raise TypeError('neighbor or weight cannot be None')
        if self.__in_window(neighbor):
            neighbor.incoming_edges += 1
            cosine_score = cosine_similarity(self.phrase.embedding,
                neighbor.phrase.embedding)
            # need to rewrite api to allow candidates to be calculated
            pmi = self.phrase.calc_pmi(neighbor.phrase, candidates)
            self.adj_weights[neighbor.key] = cosine_score * pmi
            self.adj_nodes[neighbor.key] = neighbor

    def __in_window(self, neighbor):
        window = self.phrase.window
        neighbor_pos = neighbor.phrase.positions
        for pos in neighbor_pos:
            pos0 = window.get(pos[0])
            pos1 = window.get(pos[1])
            if window.get(pos0) or window.get(pos1):
                return True
        return False

class PhraseGraph(object):
    """Bi-directional G=graph of phrases"""

    def __init__(self, candidates: List[Phrase]):
        self.nodes = {}
        self.candidates = candidates

    def add_node(self, key):
        if key is None:
            raise TypeError('key cannot be None')
        if key not in self.nodes:
            self.nodes[key] = PhraseNode(key)
        return self.nodes[key]

    def add_edge(self, source_key, dest_key, weight=0):
        if source_key is None or dest_key is None:
            raise KeyError('Invalid key')
        if source_key not in self.nodes:
            self.add_node(dest_key)
        if dest_key not in self.nodes:
            self.add_node(dest_key)
        self.nodes[source_key].add_neighbor(self.nodes[dest_key], 
            weight)