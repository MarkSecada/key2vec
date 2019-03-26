import numpy as np
from typing import Dict

class Glove(object):
    """GloVe vectors.

    Parameters
    ----------
    path : str, required
        Path to the GloVe embeddings

    Attributes
    ----------
    embeddings : Dict[str, np.float64]
        Dictionary of GloVe embeddings
    dim : int
        Dimension of GloVe embeddings
    """

    def __init__(self, path: str) -> None:
        self.embeddings = self.__read_glove(path)
        self.dim = self.__get_dim()

    def __read_glove(self, path: str) -> Dict[str, np.float64]:
        """Reads GloVe vectors into a dictionary, where
           the words are the keys, and the vectors are the values.

        Returns
        -------
        word_vectors : Dict[str, np.float64]
        """
        with open(path, 'r') as f:
            data = f.readlines()
        word_vectors = {}
        for row in data:
            stripped_row = row.strip('\n')
            split_row = stripped_row.split(' ')
            word = split_row[0]
            vector = []
            for el in split_row[1:]:
                vector.append(float(el))
            word_vectors[word] = np.array(vector)
        return word_vectors

    def __get_dim(self) -> int:
        return len(self.embeddings[list(self.embeddings.keys())[0]])