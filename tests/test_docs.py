import pytest
from key2vec.glove import Glove
from key2vec.docs import Document, Phrase

glove = Glove('../data/glove.6B/glove.6B.50d.txt')

def test_document():
    text = "Hello! My name is Mark Secada. I'm a Data Scientist."
    doc = Document(text, glove)
    assert doc.text == text
    assert doc.dim == 50
    assert doc.embedding is not None

def test_phrase():
    text = "Hello! My name is Mark Secada. I'm a Data Scientist."
    doc = Document(text, glove)
    phrase = Phrase("Mark Secada", glove, doc)
    assert phrase.text == "Mark Secada"
    assert phrase.dim == 50
    assert phrase.embedding is not None
    assert phrase.parent.text == text
    assert phrase.parent.dim == phrase.dim
    assert phrase.parent.embedding is not None
    assert type(phrase.similarity) == float

    phrase = Phrase("Secada", glove, doc)
    assert phrase.similarity == -1

