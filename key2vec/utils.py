import numpy as np
from typing import Dict

def read_glove(path: str) -> Dict[str, np.float64]:
    """Reads GloVe vectors into a dictionary, where
       the words are the keys, and the vectors are the values.

    Parameters
    ----------
    dim : int, required
        Vector size to import. Must be 50, 100, 200, or 300.

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