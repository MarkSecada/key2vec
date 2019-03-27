import pytest
from key2vec.glove import Glove

def test_glove():
    path = '../data/glove.6B/glove.6B.50d.txt'
    glove = Glove(path)
    assert glove.dim == 50
    assert glove.embeddings.get('the', None) is not None